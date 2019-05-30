using System;
using System.Collections.Generic;
using BlurredLines.Calculation;
using FluentAssertions;
using Xunit;

namespace BlurredLines.Tests.Calculation
{
    [Trait("Category", "Unit Tests")]
    public class GaussianBlurKernelCalculatorTests
    {
        private GaussianBlurKernelCalculator sut;

        public GaussianBlurKernelCalculatorTests()
        {
            sut = new GaussianBlurKernelCalculator();
        }

        [Theory]
        [MemberData(nameof(ExpectedOneDimensionalKernelValues))]
        public void ItShallCalculateCorrectOneDimensionalKernelValues(int kernelSize, double[] expectedKernel)
        {
            // When
            var actualKernelValues = sut.CalculateOneDimensionalKernel(kernelSize);

            // Then
            actualKernelValues.Length.Should().Be(expectedKernel.Length);
            for (int i = 0; i < actualKernelValues.Length; i++)
            {
                actualKernelValues[i].Should().BeApproximately(expectedKernel[i], 0.0001);
            }
        }

        [Theory]
        [InlineData(-1)]
        [InlineData(0)]
        [InlineData(2)]
        [InlineData(8)]
        public void ItShallThrowAnExceptionForInvalidKernelSizes(int kernelSize)
        {
            // When
            Action action = () => sut.CalculateOneDimensionalKernel(kernelSize);
            
            // Then
            action.Should().Throw<ArgumentOutOfRangeException>();
        }

        public static IEnumerable<object[]> ExpectedOneDimensionalKernelValues =>
            new List<object[]>
            {
                new object[] {1, new[] {1d}},
                new object[] {3, new[] {0.27901, 0.44198, 0.27901}},
                new object[] {5, new[] {0.06136, 0.24477, 0.38774, 0.24477, 0.06136}},
                new object[] {7, new[] {0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598}},
                new object[]
                {
                    9, new[] {0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229}
                },
            };
    }
}