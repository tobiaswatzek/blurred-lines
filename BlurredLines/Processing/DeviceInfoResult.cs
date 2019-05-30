namespace BlurredLines.Processing
{
    public class DeviceInfoResult
    {
        public string Name { get; set; }
        public string Vendor { get; set; }
        public string DriverVersion { get; set; }
        public string Profile { get; set; }
        public string Version { get; set; }
        public string Extensions { get; set; }
        public int MaxWorkItemDimensions { get; set; }
        public int MaxWorkGroupSize { get; set; }
        public int MaxWorkItemSizes { get; set; }

        public override string ToString()
        {
            return
                $"{nameof(Name)}: {Name},\n" +
                $"{nameof(Vendor)}: {Vendor},\n" +
                $"{nameof(DriverVersion)}: {DriverVersion},\n" +
                $"{nameof(Profile)}: {Profile},\n" +
                $"{nameof(Version)}: {Version},\n" +
                $"{nameof(Extensions)}: {Extensions},\n" +
                $"{nameof(MaxWorkItemDimensions)}: {MaxWorkItemDimensions},\n" +
                $"{nameof(MaxWorkGroupSize)}: {MaxWorkGroupSize},\n" +
                $"{nameof(MaxWorkItemSizes)}: {MaxWorkItemSizes}";
        }
    }
}