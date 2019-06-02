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
            return CommandLine.Parser.Default.ParseArguments<BlurOptions, InfoOptions>(args)
                .MapResult((BlurOptions opts) => new BlurCommand(opts).Run(),
                    (InfoOptions opts) => new InfoCommand(opts).Run(),
                    errs => 1);
        }
    }
}
