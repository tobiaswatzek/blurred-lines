using System;
using System.IO;
using BlurredLines.Processing;
using Newtonsoft.Json;

namespace BlurredLines.Commands
{
    public class InfoCommand
    {
        public int Run(InfoOptions options)
        {
            Console.WriteLine($"Running info command with out file path '{options.OutFilePath}'.");

            Console.WriteLine("Gathering system information.");
            var platformInfos = SystemInformation.GetPlatformInfos();
            Console.WriteLine("Gathered all system information. Writing to file.");
            using (var file = File.CreateText(Path.GetFullPath(options.OutFilePath)))
            {
                var serializer = new JsonSerializer {Formatting = Formatting.Indented};
                serializer.Serialize(file, platformInfos);
            }

            Console.WriteLine(
                $"System information was successfully written to '{Path.GetFullPath(options.OutFilePath)}'.");

            return 0;
        }
    }
}