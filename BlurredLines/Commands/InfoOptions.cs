using CommandLine;

namespace BlurredLines.Commands
{
    [Verb("info", HelpText = "Create a JSON file with OpenCL infos of the system.")]
    public class InfoOptions : BaseOptions
    {
        [Option('o', "out", Required = true, HelpText = "Output filepath where the JSON should be written to.")]
        public string OutFilePath { get; set; }
    }
}