using System;
using System.IO;
using SixLabors.ImageSharp;

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

            Console.WriteLine($"Loading image {options.InFilePath}.");
            using (var image = Image.Load(options.InFilePath))
            {
                Console.WriteLine(
                    $"Loaded image {options.InFilePath} with width {image.Width} and height {image.Height}.");
            }

            return 0;
        }
    }
}