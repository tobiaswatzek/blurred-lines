namespace BlurredLines.Processing.Info
{
    /// <summary>
    ///     Information about a device.
    /// </summary>
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
        public ulong MaxConstantBufferSize { get; set; }
    }
}
