using Serilog;
using Serilog.Core;
using Serilog.Sinks.SystemConsole.Themes;

namespace BlurredLines.Commands
{
    public class BaseCommand
    {
        protected readonly Logger logger;

        protected BaseCommand(BaseOptions options)
        {
            var loggerConfig = new LoggerConfiguration()
                .WriteTo.Console(theme: AnsiConsoleTheme.Code)
                .WriteTo.File("logs/blurred-lines.log", rollingInterval: RollingInterval.Day);

            if (options.Verbose)
            {
                loggerConfig.MinimumLevel.Debug();
            }
            else
            {
                loggerConfig.MinimumLevel.Information();
            }

            logger = loggerConfig.CreateLogger();
        }
    }
}
