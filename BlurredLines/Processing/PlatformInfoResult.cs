using System.Collections.Generic;

namespace BlurredLines.Processing
{
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