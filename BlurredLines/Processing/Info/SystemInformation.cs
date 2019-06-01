using System;
using System.Collections.Generic;
using System.Linq;
using OpenCL.Net;

namespace BlurredLines.Processing
{
    /// <summary>
    ///     Retrieve basic system information of the OpenCL platforms and devices.
    /// </summary>
    public static class SystemInformation
    {
        /// <summary>
        ///     Retrieve information about all platforms and devices of the system.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static IEnumerable<PlatformInfoResult> GetPlatformInfos()
        {
            var platforms = Cl.GetPlatformIDs(out var error);
            error.ThrowOnError();

            if (!platforms.Any())
            {
                throw new InvalidOperationException("No OpenCL platform available.");
            }

            return platforms.Select(GetPlatformInfo);
        }

        /// <summary>
        ///     Retrieve information about the given <see cref="Platform"/>.
        /// </summary>
        /// <param name="platform"></param>
        /// <returns></returns>
        public static PlatformInfoResult GetPlatformInfo(Platform platform)
        {
            var info = new PlatformInfoResult
            {
                Name = GetPlatformInfoPart(platform, PlatformInfo.Name),
                Version = GetPlatformInfoPart(platform, PlatformInfo.Version),
                Vendor = GetPlatformInfoPart(platform, PlatformInfo.Vendor),
                Extensions = GetPlatformInfoPart(platform, PlatformInfo.Extensions),
                Profile = GetPlatformInfoPart(platform, PlatformInfo.Profile)
            };

            var devices = Cl.GetDeviceIDs(platform, DeviceType.All, out var error);
            error.ThrowOnError();
            if (devices.Length == 0)
            {
                return info;
            }

            info.Devices = devices.Select(GetDeviceInfo);
            return info;
        }


        /// <summary>
        ///     Retrieve information about the given <see cref="Device"/>.
        /// </summary>
        /// <param name="device"></param>
        /// <returns></returns>
        public static DeviceInfoResult GetDeviceInfo(Device device)
        {
            return new DeviceInfoResult
            {
                Name = GetDeviceInfoPartString(device, DeviceInfo.Name),
                Vendor = GetDeviceInfoPartString(device, DeviceInfo.Vendor),
                DriverVersion = GetDeviceInfoPartString(device, DeviceInfo.DriverVersion),
                Profile = GetDeviceInfoPartString(device, DeviceInfo.Profile),
                Version = GetDeviceInfoPartString(device, DeviceInfo.Version),
                Extensions = GetDeviceInfoPartString(device, DeviceInfo.Extensions),
                MaxWorkGroupSize = GetDeviceInfoPartInt(device, DeviceInfo.MaxWorkGroupSize),
                MaxWorkItemSizes = GetDeviceInfoPartInt(device, DeviceInfo.MaxWorkItemSizes),
                MaxWorkItemDimensions = GetDeviceInfoPartInt(device, DeviceInfo.MaxWorkItemDimensions)
            };
        }


        private static string GetPlatformInfoPart(Platform platform, PlatformInfo part)
        {
            using (var info = Cl.GetPlatformInfo(platform, part, out var errorCode))
            {
                errorCode.ThrowOnError();
                return info.ToString();
            }
        }

        public static T GetDeviceInfoPart<T>(Device device, DeviceInfo part, Func<InfoBuffer, T> toValue)
        {
            using (var info = Cl.GetDeviceInfo(device, part, out var errorCode))
            {
                errorCode.ThrowOnError();
                return toValue(info);
            }
        }

        private static string GetDeviceInfoPartString(Device device, DeviceInfo part)
        {
            return GetDeviceInfoPart(device, part, buffer => buffer.ToString());
        }

        private static int GetDeviceInfoPartInt(Device device, DeviceInfo part)
        {
            return GetDeviceInfoPart(device, part, buffer => buffer.CastTo<int>());
        }
    }
}