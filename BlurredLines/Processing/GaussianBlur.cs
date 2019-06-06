using BlurredLines.Calculation;
using BlurredLines.Processing.Info;
using OpenCL.Net;
using Serilog.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Image = SixLabors.ImageSharp.Image;

namespace BlurredLines.Processing
{
    /// <summary>
    ///     Class to apply a gaussian blur filter to an image.
    /// </summary>
    public class GaussianBlur
    {
        private readonly Logger logger;
        private readonly GaussianBlurKernelCalculator kernelCalculator;

        public GaussianBlur(Logger logger)
        {
            this.logger = logger;
            kernelCalculator = new GaussianBlurKernelCalculator();
        }

        /// <summary>
        ///     Apply a gaussian blur filter with the given <see cref="gaussKernelSize"/> and <see cref="sigma"/> to the given <see cref="image"/>.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="gaussKernelSize"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public Image<Rgb24> Apply(Image<Rgb24> image, int gaussKernelSize, double sigma)
        {
            var gaussianKernel = kernelCalculator.CalculateOneDimensionalKernel(gaussKernelSize, sigma);

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
            using (var gaussianKernelBuffer = CreateBufferOrThrow<float>(context, MemFlags.ReadOnly, gaussKernelSize))
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
                        gaussKernelSize,
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
                    var gaussRadiusSize = gaussKernelSize >> 1;
                    logger.Debug("Building OpenCL program.");
                    BuildProgramOrThrow(program,
                        1,
                        new[] {device},
                        $"-D IMAGE_WIDTH={image.Width}",
                        $"-D IMAGE_HEIGHT={image.Height}",
                        $"-D GAUSS_KERNEL_SIZE={gaussKernelSize}",
                        $"-D GAUSS_RADIUS_SIZE={gaussRadiusSize}");
                    logger.Debug("Successfully built OpenCL program.");

                    using (var kernel = CreateKernelOrThrow(program, "gaussianBlur"))
                    {
                        logger.Debug("Successfully created OpenCL kernel");


                        logger.Debug("Setting OpenCL kernel arguments.");
                        Cl.SetKernelArg(kernel, 0, redPixelsBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 1, greenPixelsBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 2, bluePixelsBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 3, redPixelsOutBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 4, greenPixelsOutBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 5, bluePixelsOutBuffer).ThrowOnError();
                        Cl.SetKernelArg(kernel, 6, gaussianKernelBuffer).ThrowOnError();
                        logger.Debug("Successfully set OpenCL kernel arguments");


                        var maxWorkItemSizes = GetMaxWorkItemSizes(device);
                        logger.Information("Retrieved max work item sizes {MaxWorkItemSizes}.",
                            maxWorkItemSizes);

                        var maxWorkGroupSize = GetMaxWorkGroupSize(device);
                        logger.Information("Retrieved max work group size {MaxWorkGroupSize}.", maxWorkGroupSize);

                        var localSize = GetLocalSize(maxWorkGroupSize, kernel, device);

                        logger.Information(
                            "Retrieved local size {LocalSize}.",
                            localSize);

                        var verticalLocalSize = localSize;
                        var horizontalLocalSize = localSize;
                        var numberOfVerticalBatches = (image.Height + maxWorkItemSizes.y - 1) / maxWorkItemSizes.y;
                        var numberOfHorizontalBatches = (image.Width + maxWorkItemSizes.x - 1) / maxWorkItemSizes.x;
                        var totalNumberOfBatches = numberOfVerticalBatches * numberOfHorizontalBatches;
                        var kernelEvents = new List<Event>();

                        for (int y = 0; y < numberOfVerticalBatches; y++)
                        {
                            var verticalOffset = new IntPtr(y * maxWorkItemSizes.y);
                            logger.Debug("Vertical offset is {VerticalOffset}.", verticalOffset.ToInt32());
                            for (int x = 0; x < numberOfHorizontalBatches; ++x)
                            {
                                var horizontalOffset = new IntPtr(x * maxWorkItemSizes.x);
                                logger.Debug("Horizontal offset is {HorizontalOffset}.", horizontalOffset.ToInt32());

                                logger.Debug("Running batch number {VerticalBatchNumber}x{HorizontalBatchNumber} of " +
                                             "{TotalNumberOfBatches} with {NumberOfItemsX}x{NumberOfItemsY} items.",
                                    y,
                                    x,
                                    totalNumberOfBatches,
                                    maxWorkItemSizes.x,
                                    maxWorkItemSizes.y);
                                // set the kernel arguments
                                logger.Debug("Setting OpenCL local pixels kernel arguments.");
                                Cl.SetKernelArg(kernel,
                                        7,
                                        // Combine the size of the work group with padding on the horizontal and vertical sides
                                        // and multiply it by the size of float3 to allocate enough memory for the workgroup
                                        new IntPtr((verticalLocalSize + gaussKernelSize + 3) *
                                                   (horizontalLocalSize + gaussKernelSize + 3) *
                                                   sizeof(float) *
                                                   3),
                                        null)
                                    .ThrowOnError();
                                logger.Debug("Successfully set OpenCL local pixels kernel arguments.");

                                logger.Debug("Enqueuing OpenCL kernel.");
                                // execute the kernel
                                Cl.EnqueueNDRangeKernel(commandQueue,
                                        kernel,
                                        2,
                                        new[] {horizontalOffset, verticalOffset},
                                        new[] {new IntPtr(maxWorkItemSizes.x), new IntPtr(maxWorkItemSizes.y)},
                                        new[] {new IntPtr(horizontalLocalSize), new IntPtr(verticalLocalSize)},
                                        0,
                                        null,
                                        out var kernelEvent)
                                    .ThrowOnError();
                                kernelEvents.Add(kernelEvent);
                                logger.Debug("Successfully enqueued OpenCL kernel.");
                            }
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

        private int GetLocalSize(int maxWorkGroupSize, Kernel kernel, Device device)
        {
            var preferredWorkGroupSizeMultiple = GetPreferredWorkGroupSizeMultiple(kernel, device);
            var result = (int) Math.Sqrt(preferredWorkGroupSizeMultiple);

            // Ensure that the maxWorkGroupSize is evenly divisible by the local size
            while (result > 0 && maxWorkGroupSize % result != 0)
            {
                --result;
            }

            if (result > 0)
            {
                return result;
            }

            logger.Error(
                "Could not find a work group size that divides max work group size {MaxWorkGroupSize} evenly.",
                maxWorkGroupSize);
            throw new InvalidOperationException("Could not find a work group size that divides max work group size");
        }

        private static int GetPreferredWorkGroupSizeMultiple(Kernel kernel, Device device)
        {
            int preferredWorkGroupSizeMultiple;
            // CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE == (KernelWorkGroupInfo)0x11B3
            using (var kernelWorkGroupInfo =
                Cl.GetKernelWorkGroupInfo(kernel, device, (KernelWorkGroupInfo) 0x11B3, out var error))
            {
                error.ThrowOnError();
                preferredWorkGroupSizeMultiple = kernelWorkGroupInfo.CastTo<int>();
            }

            return preferredWorkGroupSizeMultiple;
        }

        private static (int x, int y, int z) GetMaxWorkItemSizes(Device device)
        {
            Cl.GetDeviceInfo(device,
                    DeviceInfo.MaxWorkItemDimensions,
                    IntPtr.Zero,
                    InfoBuffer.Empty,
                    out var paramSize)
                .ThrowOnError();

            int dimensions;
            using (var dimensionInfoBuffer = new InfoBuffer(paramSize))
            {
                Cl.GetDeviceInfo(device,
                        DeviceInfo.MaxWorkItemDimensions,
                        paramSize,
                        dimensionInfoBuffer,
                        out paramSize)
                    .ThrowOnError();
                dimensions = dimensionInfoBuffer.CastTo<int>();
            }

            Cl.GetDeviceInfo(device,
                    DeviceInfo.MaxWorkItemSizes,
                    IntPtr.Zero,
                    InfoBuffer.Empty,
                    out paramSize)
                .ThrowOnError();
            IntPtr[] maxWorkItemSizes;
            using (var maxWorkItemSizesInfoBuffer = new InfoBuffer(paramSize))
            {
                Cl.GetDeviceInfo(device,
                        DeviceInfo.MaxWorkItemSizes,
                        paramSize,
                        maxWorkItemSizesInfoBuffer,
                        out paramSize)
                    .ThrowOnError();
                maxWorkItemSizes = maxWorkItemSizesInfoBuffer.CastToArray<IntPtr>(dimensions);
            }

            return (maxWorkItemSizes[0].ToInt32(), maxWorkItemSizes[1].ToInt32(), maxWorkItemSizes[2].ToInt32());
        }

        private static int GetMaxWorkGroupSize(Device device)
        {
            return SystemInformation.GetDeviceInfoPart(device,
                DeviceInfo.MaxWorkGroupSize,
                buffer => buffer.CastTo<int>());
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
