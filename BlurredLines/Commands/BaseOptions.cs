using CommandLine;

namespace BlurredLines.Commands
{
    public abstract class BaseOptions
    {
        [Option('v', "verbose", Required = false, Default = false, HelpText = "Be more verbose about what's happening.")]
        public bool Verbose { get; set; }
    }
}
