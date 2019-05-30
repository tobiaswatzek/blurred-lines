using System;
using System.IO;
using System.Linq;
using BlurredLines.Calculation;
using OpenCL.Net;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace BlurredLines.Processing
{
    public class GaussianBlur
    {
        private readonly GaussianBlurKernelCalculator kernelCalculator;

        public GaussianBlur()
        {
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

            const int elementSize = 200;
            const int dataSize = elementSize * sizeof(int);
            var vectorA = new int[elementSize];
            var vectorB = new int[elementSize];
            var vectorC = new int[elementSize];

            for (var i = 0; i < elementSize; i++)
            {
                vectorA[i] = i;
                vectorB[i] = i;
            }

            using (var context = CreateContextOrThrow(devices))
            using (var commandQueue = CreateCommandQueueOrThrow(context, device))
            using (var bufferA = CreateBufferOrThrow<int>(context, MemFlags.ReadOnly, dataSize))
            using (var bufferB = CreateBufferOrThrow<int>(context, MemFlags.ReadOnly, dataSize))
            using (var bufferC = CreateBufferOrThrow<int>(context, MemFlags.ReadOnly, dataSize))
            {
                Cl.EnqueueWriteBuffer(commandQueue,
                        bufferA,
                        Bool.True,
                        0,
                        dataSize,
                        vectorA,
                        0,
                        null,
                        out _)
                    .ThrowOnError();
                Cl.EnqueueWriteBuffer(commandQueue,
                        bufferB,
                        Bool.True,
                        0,
                        dataSize,
                        vectorB,
                        0,
                        null,
                        out _)
                    .ThrowOnError();
                var programSource = File.ReadAllText("Kernels/test.cl");
                using (var program = CreateProgramWithSourceOrThrow(context, programSource))
                {
                    BuildProgramOrThrow(program, 1, new[] {device});
                    using (var kernel = CreateKernelOrThrow(program, "vector_add"))
                    {
                        // set the kernel arguments
                        Cl.SetKernelArg(kernel, 0, bufferA).ThrowOnError();
                        Cl.SetKernelArg(kernel, 1, bufferB).ThrowOnError();
                        Cl.SetKernelArg(kernel, 2, bufferC).ThrowOnError();

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
                        var maxWorkItemSizeX = maxWorkItemSizes[0].ToInt32();

                        if (elementSize > maxWorkItemSizeX)
                        {
                            Console.WriteLine(
                                "Error: Too many elements to process - maximum elements allowed: " +
                                maxWorkItemSizeX);
                            System.Environment.Exit(1);
                        }

                        // execute the kernel
                        Cl.EnqueueNDRangeKernel(commandQueue,
                                kernel,
                                1,
                                null,
                                new IntPtr[] {new IntPtr(elementSize)},
                                null,
                                0,
                                null,
                                out _)
                            .ThrowOnError();

                        // read the device output buffer to the host output array
                        Cl.EnqueueReadBuffer(commandQueue,
                                bufferC,
                                Bool.True,
                                0,
                                dataSize,
                                vectorC,
                                0,
                                null,
                                out _)
                            .ThrowOnError();

                        // output result
                        PrintVector(vectorA, elementSize, "Input A");
                        PrintVector(vectorB, elementSize, "Input B");
                        PrintVector(vectorC, elementSize, "Output C");
                    }
                }
            }

            return image;
        }
        
        static void PrintVector(int[] vector, int elementSize, string label)
        {
            Console.WriteLine(label + ":");

            for (var i = 0; i < elementSize; ++i)
            {
                Console.Write(vector[i] + " ");
            }

            Console.WriteLine();
        }

        private static Kernel CreateKernelOrThrow(OpenCL.Net.Program program, string kernelName)
        {
            var kernel = Cl.CreateKernel(program, "vector_add", out var error);
            error.ThrowOnError();
            return kernel;
        }

        private static void BuildProgramOrThrow(OpenCL.Net.Program program, uint numDevices, Device[] devices)
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
            Console.Error.WriteLine($"Build Error: {infoBuffer}");
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
