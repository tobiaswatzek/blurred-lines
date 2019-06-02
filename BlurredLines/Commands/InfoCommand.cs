using System;
using System.IO;
using BlurredLines.Processing;
using BlurredLines.Processing.Info;
using Newtonsoft.Json;
using Serilog.Core;

namespace BlurredLines.Commands
{
    public class InfoCommand : BaseCommand
    {
        private readonly InfoOptions options;

        public InfoCommand(InfoOptions options) : base(options)
        {
            this.options = options;
        }

        public int Run()
        {
            logger.Information("Running info command with out file path '{OutFilePath}'.",
                options.OutFilePath);

            logger.Information("Gathering system information.");
            var platformInfos = SystemInformation.GetPlatformInfos();
            logger.Information("Gathered all system information. Writing to file.");
            using (var file = File.CreateText(Path.GetFullPath(options.OutFilePath)))
            {
                var serializer = new JsonSerializer {Formatting = Formatting.Indented};
                serializer.Serialize(file, platformInfos);
            }

            logger.Information("System information was successfully written to '{OutFilePath}'.",
                Path.GetFullPath(options.OutFilePath));

            return 0;
        }
    }
}
