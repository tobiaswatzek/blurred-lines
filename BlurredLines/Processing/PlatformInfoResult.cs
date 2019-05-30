using System.Collections.Generic;

namespace BlurredLines.Processing
{
    /// <summary>
    ///     Information about a platform.
    /// </summary>
    public class PlatformInfoResult
    {
        public string Profile { get; set; }
        public string Version { get; set; }
        public string Name { get; set; }
        public string Vendor { get; set; }
        public string Extensions { get; set; }

        public IEnumerable<DeviceInfoResult> Devices { get; set; }
    }
}