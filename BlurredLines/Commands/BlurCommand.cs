using System;
using System.IO;
using System.Linq;
using BlurredLines.Calculation;
using BlurredLines.Processing;
using Serilog.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace BlurredLines.Commands
{
    public class BlurCommand
    {
        private readonly Logger logger;

        public BlurCommand(Logger logger)
        {
            this.logger = logger;
        }

        public int Run(BlurOptions options)
        {
            logger.Information("Running blur command with in file path: '{InFilePath}', " +
                               "out file path '{OutFilePath}' " +
                               "and kernel size {KernelSize}.",
                options.InFilePath,
                options.OutFilePath,
                options.KernelSize);
            if (!File.Exists(options.InFilePath))
            {
                logger.Error("There exists no file at the given path '{InFilePath}'.", options.InFilePath);
                return 1;
            }

            if (options.KernelSize < 1 || options.KernelSize > 9)
            {
                logger.Error("Kernel size has to be between 1 and 9 and not {KernelSize}.", options.KernelSize);
                return 1;
            }

            var gaussianBlurKernelCalculator = new GaussianBlurKernelCalculator();

            var gaussianKernel = gaussianBlurKernelCalculator.CalculateOneDimensionalKernel(options.KernelSize);

            logger.Debug("Gaussian kernel is {GaussianKernel}.", gaussianKernel);

            logger.Information("Loading image {InFilePath}.", options.InFilePath);
            using (var image = Image.Load<Rgb24>(options.InFilePath))
            {
                logger.Information("Loaded image {InFilePath} with width {Width} and height {Height}.",
                    options.InFilePath,
                    image.Width,
                    image.Height);

                var gaussianBlur = new GaussianBlur(logger);
                logger.Information("Starting image blur processing.");
                var blurredImage = gaussianBlur.Apply(image, options.KernelSize);
                logger.Information("Finished image blur processing.");
                
                logger.Debug("Storing image at '{OutputImageLocation}'.", Path.GetFullPath(options.OutFilePath));
                blurredImage.Save(options.OutFilePath);
                logger.Debug("Successfully stored image at '{OutputImageLocation}'.", Path.GetFullPath(options.OutFilePath));
            }

            return 0;
        }
    }
}
