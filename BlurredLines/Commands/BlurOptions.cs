using CommandLine;

namespace BlurredLines.Commands
{
    [Verb("blur", HelpText = "Apply a gaussian blur to an image and save as new image.")]
    public class BlurOptions
    {
        [Option('i', "in", Required = true, HelpText = "Image file that should be blurred.")]
        public string InFilePath { get; set; }
        
        [Option('o', "out", Required = true, HelpText = "Path where the blurred image should be written to.")]
        public string OutFilePath { get; set; }

        [Option('k', "kernel", Default = 3, HelpText = "Kernel size that should be used.")]
        public int KernelSize { get; set; }
    }
}