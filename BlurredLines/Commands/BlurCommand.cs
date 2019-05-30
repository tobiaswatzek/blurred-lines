using System;
using System.IO;
using System.Linq;
using BlurredLines.Calculation;
using BlurredLines.Processing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace BlurredLines.Commands
{
    public class BlurCommand
    {
        public int Run(BlurOptions options)
        {
            Console.WriteLine($"Running blur command with in file path: '{options.InFilePath}', " +
                              $"out file path '{options.OutFilePath}' " +
                              $"and kernel size {options.KernelSize}.");
            if (!File.Exists(options.InFilePath))
            {
                Console.Error.WriteLine($"There exists no file at the given path '{options.InFilePath}'.");
                return 1;
            }

            if (options.KernelSize < 1 || options.KernelSize > 9)
            {
                Console.Error.WriteLine($"Kernel size has to be between 1 and 9 and not {options.KernelSize}.");
                return 1;
            }

            var gaussianBlurKernelCalculator = new GaussianBlurKernelCalculator();

            var kernel = gaussianBlurKernelCalculator.CalculateOneDimensionalKernel(options.KernelSize);

            kernel.ToList().ForEach(Console.WriteLine);

            Console.WriteLine($"Loading image {options.InFilePath}.");
            using (var image = Image.Load<Rgb24>(options.InFilePath))
            {
                Console.WriteLine(
                    $"Loaded image {options.InFilePath} with width {image.Width} and height {image.Height}.");

                var gaussianBlur = new GaussianBlur();
                var blurredImage = gaussianBlur.Apply(image, options.KernelSize);
            }

            return 0;
        }
    }
}
