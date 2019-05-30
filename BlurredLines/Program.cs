using System;
using System.IO;
using BlurredLines.Processing;
using Newtonsoft.Json;
using SixLabors.ImageSharp;

namespace BlurredLines
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.Error.WriteLine("The program expects exactly two arguments.");
                Console.Error.WriteLine("1. The path to the image.");
                Console.Error.WriteLine("2. The kernel size.");
                Environment.Exit(1);
            }

            Console.WriteLine($"File: {args[0]}");
            Console.WriteLine($"Blur: {args[1]}");

            var path = args[0];
            if (!File.Exists(path))
            {
                Console.Error.WriteLine($"There exists no file at the given path '{path}'.");
                Environment.Exit(1);
            }

            if (!int.TryParse(args[1], out var kernelSize))
            {
                Console.Error.WriteLine($"Could not parse the kernel size as number '{args[1]}'.");
                Environment.Exit(1);
            }

            if (kernelSize < 1 || kernelSize > 9)
            {
                Console.Error.WriteLine($"Please enter a kernel size between 1 and 9 and not '{kernelSize}'.");
                Environment.Exit(1);
            }


            var platformInfos = SystemInformation.GetPlatformInfos();

            using (var file = File.CreateText(Path.Combine(AppContext.BaseDirectory, "platformInfos.json")))
            {
                var serializer = new JsonSerializer {Formatting = Formatting.Indented};
                serializer.Serialize(file, platformInfos);
            }

            //            Console.WriteLine($"Loading image {path}.");
//            using (var image = Image.Load(path))
//            {
//                Console.WriteLine($"Loaded image {path} with width {image.Width} and height {image.Height}.");
//            }
        }
    }
}