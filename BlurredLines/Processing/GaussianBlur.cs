using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BlurredLines.Calculation;
using OpenCL.Net;
using Serilog.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

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
            var deviceInfo = SystemInformation.GetDeviceInfo(device);

            var pixels = image.GetPixelSpan().ToArray();
            var imageChannelDataSize = pixels.Length * sizeof(byte);
            var gaussianKernelDataSize = kernelSize * sizeof(float);

            logger.Debug("Start creating buffers.");
            using (var context = CreateContextOrThrow(devices))
            using (var commandQueue = CreateCommandQueueOrThrow(context, device))
                // input buffers
            using (var inRedImageBuffer =
                CreateBufferOrThrow<byte>(context, MemFlags.ReadOnly, imageChannelDataSize))
            using (var inGreenImageBuffer =
                CreateBufferOrThrow<byte>(context, MemFlags.ReadOnly, imageChannelDataSize))
            using (var inBlueImageBuffer =
                    CreateBufferOrThrow<byte>(context, MemFlags.ReadOnly, imageChannelDataSize))
                // output buffers
            // TODO: Use imageChannelDataSize instead of 1024
            using (var outRedImageBuffer =
                CreateBufferOrThrow<byte>(context, MemFlags.WriteOnly, 1024))
            using (var outGreenImageBuffer =
                CreateBufferOrThrow<byte>(context, MemFlags.WriteOnly, 1024))
            using (var outBlueImageBuffer =
                    CreateBufferOrThrow<byte>(context, MemFlags.WriteOnly, 1024))
                // gaussian kernel
            using (var gaussianKernelBuffer =
                CreateBufferOrThrow<float>(context, MemFlags.ReadOnly, gaussianKernelDataSize))
            {
                logger.Debug("Created buffers.");

                var writeEvents = new List<Event>();
                Cl.EnqueueWriteBuffer(commandQueue,
                        inRedImageBuffer,
                        Bool.False,
                        0,
                        imageChannelDataSize,
                        pixels.Select(pixel => pixel.R).ToArray(),
                        0,
                        null,
                        out var redWriteEvent)
                    .ThrowOnError();
                writeEvents.Add(redWriteEvent);
                Cl.EnqueueWriteBuffer(commandQueue,
                        inGreenImageBuffer,
                        Bool.False,
                        0,
                        imageChannelDataSize,
                        pixels.Select(pixel => pixel.G).ToArray(),
                        0,
                        null,
                        out var greenWriteEvent)
                    .ThrowOnError();
                writeEvents.Add(greenWriteEvent);
                Cl.EnqueueWriteBuffer(commandQueue,
                        inBlueImageBuffer,
                        Bool.True,
                        0,
                        imageChannelDataSize,
                        pixels.Select(pixel => pixel.B).ToArray(),
                        0,
                        null,
                        out var blueWriteEvent)
                    .ThrowOnError();
                writeEvents.Add(blueWriteEvent);

                Cl.EnqueueWriteBuffer(commandQueue,
                        gaussianKernelBuffer,
                        Bool.False,
                        0,
                        gaussianKernelDataSize,
                        gaussianKernel,
                        0,
                        null,
                        out var gaussianKernelEvent)
                    .ThrowOnError();
                writeEvents.Add(gaussianKernelEvent);

                Cl.WaitForEvents((uint) writeEvents.Count, writeEvents.ToArray()).ThrowOnError();

                logger.Debug("Wrote data to all buffers.");

                logger.Debug("Creating OpenCL program.");
                var programSource =
                    File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Kernels/gaussianBlur.cl"));
                using (var program = CreateProgramWithSourceOrThrow(context, programSource))
                {
                    BuildProgramOrThrow(program, 1, new[] {device});
                    logger.Debug("Successfully built OpenCL program.");
                    logger.Debug("Creating OpenCL kernel.");
                    using (var kernel = CreateKernelOrThrow(program, "gaussianBlur"))
                    {
                        logger.Debug("Successfully created OpenCL kernel");
                        // define an index space for execution
                        Cl.GetDeviceInfo(device,
                                DeviceInfo.MaxWorkItemDimensions,
                                IntPtr.Zero,
                                InfoBuffer.Empty,
                                out var paramSize)
                            .ThrowOnError();
                        var dimensionInfoBuffer = new InfoBuffer(paramSize);
                        Cl.GetDeviceInfo(device,
                                DeviceInfo.MaxWorkItemDimensions,
                                paramSize,
                                dimensionInfoBuffer,
                                out paramSize)
                            .ThrowOnError();
                        var dimensions = dimensionInfoBuffer.CastTo<int>();

                        Cl.GetDeviceInfo(device,
                                DeviceInfo.MaxWorkItemSizes,
                                IntPtr.Zero,
                                InfoBuffer.Empty,
                                out paramSize)
                            .ThrowOnError();
                        var maxWorkItemSizesInfoBuffer = new InfoBuffer(paramSize);
                        Cl.GetDeviceInfo(device,
                                DeviceInfo.MaxWorkItemSizes,
                                paramSize,
                                maxWorkItemSizesInfoBuffer,
                                out paramSize)
                            .ThrowOnError();
                        var maxWorkItemSizes =
                            maxWorkItemSizesInfoBuffer.CastToArray<IntPtr>(dimensions);
                        var maxWorkItemSizeRed = maxWorkItemSizes[0].ToInt32();
                        var maxWorkItemSizeGreen = maxWorkItemSizes[1].ToInt32();
                        var maxWorkItemSizeBlue = maxWorkItemSizes[2].ToInt32();

                        logger.Information(
                            "Retrieved work item sizes red {WorkItemSizeRed}, green {WorkItemSizeGreen} and blue {WorkItemSizeBlue}.",
                            maxWorkItemSizeRed,
                            maxWorkItemSizeGreen,
                            maxWorkItemSizeBlue);
//                        __global const unsigned char *red,
//                            __global const unsigned char *green,
//                            __global const unsigned char *blue,
//                            __global const unsigned char *redOut,
//                            __global const unsigned char *greenOut,
//                            __global const unsigned char *blueOut,
//                            __local unsigned char *redLocal,
//                            __local unsigned char *greenLocal,
//                            __local unsigned char *blueLocal,
//                            __global const double *kernel,
//                        int kernelSize,
//                        int width)

                        logger.Debug("Setting OpenCL kernel arguments.");
                        // set the kernel arguments
                        Cl.SetKernelArg(kernel, 0, inRedImageBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 1, inGreenImageBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 2, inBlueImageBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 3, outRedImageBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 4, outGreenImageBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 5, outBlueImageBuffer).ThrowOnError();
                        Cl.SetKernelArg<byte>(kernel, 6, maxWorkItemSizeRed).ThrowOnError();
                        Cl.SetKernelArg<byte>(kernel, 7, maxWorkItemSizeRed).ThrowOnError();
                        Cl.SetKernelArg<byte>(kernel, 8, maxWorkItemSizeRed).ThrowOnError();
                        Cl.SetKernelArg(kernel, 9, gaussianKernelBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 10, kernelSize);
                        Cl.SetKernelArg(kernel, 11, image.Width);
        
                        logger.Debug("Successfully set OpenCL kernel arguments");
                        
                        logger.Debug("Running OpenCL kernel.");
                        // execute the kernel
                        Cl.EnqueueNDRangeKernel(commandQueue,
                                kernel,
                                3,
                                null,
                                maxWorkItemSizes,
                                null,
                                0,
                                null,
                                out _)
                            .ThrowOnError();
                        logger.Debug("Successfully ran OpenCL kernel.");
                        // read the device output buffer to the host output array
                        var readEvents = new List<Event>();
                        var outRed = new byte[maxWorkItemSizeRed];
                        var outGreen = new byte[maxWorkItemSizeGreen];
                        var outBlue = new byte[maxWorkItemSizeBlue];
                        
                        logger.Debug("Reading result from OpenCL buffers.");
                        Cl.EnqueueReadBuffer(commandQueue,
                            outRedImageBuffer,
                            Bool.False,
                            0,
                            maxWorkItemSizeRed,
                            outRed,
                            0,
                            null,
                            out var redReadEvent);
                        readEvents.Add(redReadEvent);

                        Cl.EnqueueReadBuffer(commandQueue,
                            outGreenImageBuffer,
                            Bool.False,
                            0,
                            maxWorkItemSizeGreen,
                            outGreen,
                            0,
                            null,
                            out var greenReadEvent);
                        readEvents.Add(greenReadEvent);

                        Cl.EnqueueReadBuffer(commandQueue,
                            outBlueImageBuffer,
                            Bool.False,
                            0,
                            maxWorkItemSizeBlue,
                            outBlue,
                            0,
                            null,
                            out var blueReadEvent);
                        readEvents.Add(blueReadEvent);

                        Cl.WaitForEvents((uint) readEvents.Count, readEvents.ToArray());
                        logger.Debug("Successfully read result from OpenCL buffers.");
    
                        var modifiedPixels = Enumerable.Range(0, maxWorkItemSizeRed)
                            .Select(index => new Rgb24(outRed[index], outGreen[index], outBlue[index]))
                            .ToArray();
                        logger.Debug("Successfully created pixels from the color channels.");
                        
                        logger.Debug("Creating image from pixels.");
                        var modifiedImage = Image.LoadPixelData(modifiedPixels, (int) Math.Sqrt(maxWorkItemSizeRed), (int) Math.Sqrt(maxWorkItemSizeRed));
                        logger.Debug("Successfully created image.");
                        
                        return modifiedImage;
                    }
                }
            }
        }

        private static Kernel CreateKernelOrThrow(OpenCL.Net.Program program, string kernelName)
        {
            var kernel = Cl.CreateKernel(program, kernelName, out var error);
            error.ThrowOnError();
            return kernel;
        }

        private void BuildProgramOrThrow(OpenCL.Net.Program program, uint numDevices, Device[] devices)
        {
            var error = Cl.BuildProgram(program, numDevices, devices, "", null, IntPtr.Zero);
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
