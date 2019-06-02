using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using BlurredLines.Calculation;
using BlurredLines.Processing.Info;
using OpenCL.Net;
using Serilog.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;

namespace BlurredLines.Processing
{
    public class GaussianBlur
    {
        private readonly Logger logger;
        private readonly GaussianBlurKernelCalculator kernelCalculator;

        public GaussianBlur(Logger logger)
        {
            this.logger = logger;
            kernelCalculator = new GaussianBlurKernelCalculator();
        }

        public Image<Rgb24> Apply(Image<Rgb24> image, int kernelSize)
        {
            var gaussianKernel = kernelCalculator.CalculateOneDimensionalKernel(kernelSize);

            var platforms = Cl.GetPlatformIDs(out var error);
            error.ThrowOnError();

            if (!platforms.Any())
            {
                throw new InvalidOperationException("No OpenCL platforms found.");
            }

            var devices = Cl.GetDeviceIDs(platforms.First(), DeviceType.All, out error);
            error.ThrowOnError();

            if (!devices.Any())
            {
                throw new InvalidOperationException("No OpenCL devices found.");
            }

            var device = devices.First();


            var pixels = image.GetPixelSpan()
                .ToArray();

            logger.Debug("Start creating context, command queue and buffers.");
            using (var context = CreateContextOrThrow(devices))
            using (var commandQueue = CreateCommandQueueOrThrow(context, device))
                // input buffers
            using (var redPixelsBuffer = CreateBufferOrThrow<byte>(context, MemFlags.ReadOnly, pixels.Length))
            using (var greenPixelsBuffer = CreateBufferOrThrow<byte>(context, MemFlags.ReadOnly, pixels.Length))
            using (var bluePixelsBuffer = CreateBufferOrThrow<byte>(context, MemFlags.ReadOnly, pixels.Length))
                // output buffers
            using (var redPixelsOutBuffer = CreateBufferOrThrow<byte>(context, MemFlags.WriteOnly, pixels.Length))
            using (var greenPixelsOutBuffer = CreateBufferOrThrow<byte>(context, MemFlags.WriteOnly, pixels.Length))
            using (var bluePixelsOutBuffer = CreateBufferOrThrow<byte>(context, MemFlags.WriteOnly, pixels.Length))
                // gaussian kernel
            using (var gaussianKernelBuffer = CreateBufferOrThrow<float>(context, MemFlags.ReadOnly, kernelSize))
            {
                logger.Debug("Created context, command queue and buffers.");

                var writeEvents = new List<Event>();
                Cl.EnqueueWriteBuffer(commandQueue,
                        redPixelsBuffer,
                        Bool.False,
                        0,
                        pixels.Length,
                        pixels.Select(pixel => pixel.R).ToArray(),
                        0,
                        null,
                        out var redPixelsBufferWriteEvent)
                    .ThrowOnError();
                writeEvents.Add(redPixelsBufferWriteEvent);

                Cl.EnqueueWriteBuffer(commandQueue,
                        greenPixelsBuffer,
                        Bool.False,
                        0,
                        pixels.Length,
                        pixels.Select(pixel => pixel.G).ToArray(),
                        0,
                        null,
                        out var greenPixelsBufferWriteEvent)
                    .ThrowOnError();
                writeEvents.Add(greenPixelsBufferWriteEvent);

                Cl.EnqueueWriteBuffer(commandQueue,
                        bluePixelsBuffer,
                        Bool.False,
                        0,
                        pixels.Length,
                        pixels.Select(pixel => pixel.B).ToArray(),
                        0,
                        null,
                        out var bluePixelsBufferWriteEvent)
                    .ThrowOnError();
                writeEvents.Add(bluePixelsBufferWriteEvent);

                Cl.EnqueueWriteBuffer(commandQueue,
                        gaussianKernelBuffer,
                        Bool.False,
                        0,
                        kernelSize,
                        gaussianKernel,
                        0,
                        null,
                        out var gaussianKernelEvent)
                    .ThrowOnError();
                writeEvents.Add(gaussianKernelEvent);

                Cl.WaitForEvents((uint) writeEvents.Count, writeEvents.ToArray()).ThrowOnError();

                logger.Debug("Wrote data to all buffers.");

                logger.Debug("Creating OpenCL program.");
                var programSource = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Kernels/gaussianBlur.cl"));
                using (var program = CreateProgramWithSourceOrThrow(context, programSource))
                {
                    logger.Debug("Building OpenCL program.");
                    BuildProgramOrThrow(program,
                        1,
                        new[] {device},
                        $"-D IMAGE_WIDTH={image.Width}",
                        $"-D IMAGE_HEIGHT={image.Height}",
                        $"-D GAUSS_KERNEL_SIZE={kernelSize}");
                    logger.Debug("Successfully built OpenCL program.");

                    using (var kernel = CreateKernelOrThrow(program, "gaussianBlur"))
                    {
                        logger.Debug("Successfully created OpenCL kernel");


                        var maxWorkItemSize = GetMaxWorkItemSize(device);

                        logger.Information("Retrieved max work item size {MaxWorkItemSize}.",
                            maxWorkItemSize);
//            __constant const unsigned char *redPixels,
//                __constant const unsigned char *greenPixels,
//                __constant const unsigned char *bluePixels,
//                __global unsigned char *redPixelsOut,
//                __global unsigned char *greenPixelsOut,
//                __global unsigned char *bluePixelsOut,
//                __local unsigned char *redPixelsLocal,
//                __local unsigned char *greenPixelsLocal,
//                __local unsigned char *bluePixelsLocal,
//                __constant  const float *gaussianKernel,
//            int kernelSize,
//            int width)
                        logger.Debug("Setting OpenCL kernel arguments.");
                        Cl.SetKernelArg(kernel, 0, redPixelsBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 1, greenPixelsBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 2, bluePixelsBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 3, redPixelsOutBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 4, greenPixelsOutBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 5, bluePixelsOutBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 6, gaussianKernelBuffer).ThrowOnError();
                        logger.Debug("Successfully set OpenCL kernel arguments");

                        var numberOfBatches = (pixels.Length + maxWorkItemSize - 1) / maxWorkItemSize;
                        var kernelEvents = new List<Event>(numberOfBatches);
                        for (int i = 0; i < numberOfBatches; ++i)
                        {
                            var offset = new IntPtr(i * maxWorkItemSize);

                            var numberOfRemainingPixels = pixels.Length - (i * maxWorkItemSize);
                            var numberOfWorkItemsVal = numberOfRemainingPixels >= maxWorkItemSize
                                ? maxWorkItemSize
                                : numberOfRemainingPixels;
                            var numberOfWorkItems = new IntPtr(numberOfWorkItemsVal);

                            logger.Debug(
                                "Running batch number {BatchNumber} of {BatchTotal} with {NumberOfItems} items.",
                                i,
                                numberOfBatches,
                                numberOfWorkItemsVal);
                            // set the kernel arguments
                            logger.Debug("Setting OpenCL local pixels kernel arguments.");
                            Cl.SetKernelArg(kernel, 7, new IntPtr(numberOfWorkItemsVal * sizeof(float) * 3), null ).ThrowOnError();
                            logger.Debug("Successfully set OpenCL local pixels kernel arguments.");

                            logger.Debug("Enqueuing OpenCL kernel.");
                            // execute the kernel
                            Cl.EnqueueNDRangeKernel(commandQueue,
                                    kernel,
                                    1,
                                    new[] {offset},
                                    new[] {numberOfWorkItems},
                                    null,
                                    0,
                                    null,
                                    out var kernelEvent)
                                .ThrowOnError();
                            kernelEvents.Add(kernelEvent);
                            logger.Debug("Successfully enqueued OpenCL kernel.");
                        }

                        logger.Debug("Waiting for {NumberOfKernelEvents} kernels to complete.", kernelEvents.Count);
                        Cl.WaitForEvents((uint) kernelEvents.Count, kernelEvents.ToArray())
                            .ThrowOnError();
                        logger.Debug("Kernel events finished.");

                        // read the device output buffer to the host output array
                        var readEvents = new List<Event>();
                        var redPixelsOut = new byte[pixels.Length];
                        var greenPixelsOut = new byte[pixels.Length];
                        var bluePixelsOut = new byte[pixels.Length];
                        logger.Debug("Reading result from OpenCL buffers.");
                        Cl.EnqueueReadBuffer(commandQueue,
                                redPixelsOutBuffer,
                                Bool.False,
                                0,
                                pixels.Length,
                                redPixelsOut,
                                0,
                                null,
                                out var redPixelsBufferReadEvent)
                            .ThrowOnError();
                        readEvents.Add(redPixelsBufferReadEvent);
                        Cl.EnqueueReadBuffer(commandQueue,
                                greenPixelsOutBuffer,
                                Bool.False,
                                0,
                                pixels.Length,
                                greenPixelsOut,
                                0,
                                null,
                                out var greenPixelsBufferReadEvent)
                            .ThrowOnError();
                        readEvents.Add(greenPixelsBufferReadEvent);
                        Cl.EnqueueReadBuffer(commandQueue,
                                bluePixelsOutBuffer,
                                Bool.False,
                                0,
                                pixels.Length,
                                bluePixelsOut,
                                0,
                                null,
                                out var bluePixelsBufferReadEvent)
                            .ThrowOnError();
                        readEvents.Add(bluePixelsBufferReadEvent);

                        Cl.WaitForEvents((uint) readEvents.Count, readEvents.ToArray());
                        logger.Debug("Successfully read result from OpenCL buffers.");

                        logger.Debug("Converting data to pixels.");

                        var modifiedPixels = Enumerable.Range(0, pixels.Length)
                            .Select(i => new Rgb24(redPixelsOut[i], greenPixelsOut[i], bluePixelsOut[i]));
                        logger.Debug("Successfully converted data to pixels.");

                        logger.Debug("Creating image from pixels.");
                        var modifiedImage = Image.LoadPixelData(modifiedPixels.ToArray(), image.Width, image.Height);
                        logger.Debug("Successfully created image.");

                        return modifiedImage;
                    }
                }
            }
        }

        private static int GetMaxWorkItemSize(Device device)
        {
            return SystemInformation.GetDeviceInfoPart(device,
                DeviceInfo.MaxWorkItemSizes,
                buffer =>
                {
                    var maxWorkItemSizes = buffer.CastToArray<IntPtr>(1);
                    return maxWorkItemSizes[0].ToInt32();
                });
        }

        private static Kernel CreateKernelOrThrow(OpenCL.Net.Program program, string kernelName)
        {
            var kernel = Cl.CreateKernel(program, kernelName, out var error);
            error.ThrowOnError();
            return kernel;
        }

        private void BuildProgramOrThrow(OpenCL.Net.Program program,
            uint numDevices,
            Device[] devices,
            params string[] options)
        {
            var error = Cl.BuildProgram(program, numDevices, devices, string.Join(" ", options), null, IntPtr.Zero);
            if (error == ErrorCode.Success)
            {
                return;
            }

            var infoBuffer = Cl.GetProgramBuildInfo(program,
                devices.First(),
                ProgramBuildInfo.Log,
                out error);
            error.ThrowOnError();
            logger.Error("OpenCL kernel build error: {ErrorMessage}.", infoBuffer);
            throw new InvalidOperationException("Could not build OpenCL program.");
        }

        private static OpenCL.Net.Program CreateProgramWithSourceOrThrow(Context context, string programSource)
        {
            var program = Cl.CreateProgramWithSource(context, 1, new string[] {programSource}, null, out var error);
            error.ThrowOnError();
            return program;
        }

        private static IMem<T> CreateBufferOrThrow<T>(Context context, MemFlags memFlags, int length)
            where T : struct
        {
            var buffer = Cl.CreateBuffer<T>(context, memFlags, length, out var error);
            error.ThrowOnError();
            return buffer;
        }

        private static CommandQueue CreateCommandQueueOrThrow(Context context, Device device)
        {
            var commandQueue = Cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out var error);
            error.ThrowOnError();
            return commandQueue;
        }

        private static Context CreateContextOrThrow(Device[] devices)
        {
            var context = Cl.CreateContext(null, 1, devices, null, IntPtr.Zero, out var error);
            error.ThrowOnError();
            return context;
        }
    }
}
