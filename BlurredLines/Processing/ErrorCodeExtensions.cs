using System;
using OpenCL.Net;

namespace BlurredLines.Processing
{
    /// <summary>
    ///     Extensions for <see cref="ErrorCode"/>
    /// </summary>
    public static class ErrorCodeExtensions
    {
        /// <summary>
        ///     Check if the <see cref="errorCode"/> is <see cref="ErrorCode.Success"/> and if not throw an exception.
        /// </summary>
        /// <param name="errorCode"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public static void ThrowOnError(this ErrorCode errorCode)
        {
            if (errorCode == ErrorCode.Success)
            {
                return;
            }

            throw new InvalidOperationException("An OpenCL error occured. " +
                                                $"The status code has the value {errorCode:D} " +
                                                $"which corresponds to {errorCode:G}.");
        }
    }
}