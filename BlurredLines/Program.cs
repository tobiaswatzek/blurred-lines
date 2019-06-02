using System;
using System.IO;
using BlurredLines.Commands;
using BlurredLines.Processing;
using CommandLine;
using Newtonsoft.Json;
using Serilog;
using Serilog.Sinks.SystemConsole.Themes;

namespace BlurredLines
{
    internal class Program
    {
        public static int Main(string[] args)
        {
            var logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.Console(theme: AnsiConsoleTheme.Code)
                .WriteTo.File("logs/blurred-lines.log", rollingInterval: RollingInterval.Day)
                .CreateLogger();

            logger.Information("Starting program.");

            return CommandLine.Parser.Default.ParseArguments<BlurOptions, InfoOptions>(args)
                .MapResult((BlurOptions opts) => new BlurCommand(logger).Run(opts),
                    (InfoOptions opts) => new InfoCommand(logger).Run(opts),
                    errs => 1);
        }
    }
}
