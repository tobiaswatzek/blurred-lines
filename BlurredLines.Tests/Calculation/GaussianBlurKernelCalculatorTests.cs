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
        public void ItShallCalculateCorrectOneDimensionalKernelValues(int kernelSize, float[] expectedKernel)
        {
            // When
            var actualKernelValues = sut.CalculateOneDimensionalKernel(kernelSize, 1);

            // Then
            actualKernelValues.Length.Should().Be(expectedKernel.Length);
            for (int i = 0; i < actualKernelValues.Length; i++)
            {
                actualKernelValues[i].Should().BeApproximately(expectedKernel[i], 0.0001f);
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
            Action action = () => sut.CalculateOneDimensionalKernel(kernelSize, 1);
            
            // Then
            action.Should().Throw<ArgumentOutOfRangeException>();
        }

        public static IEnumerable<object[]> ExpectedOneDimensionalKernelValues =>
            new List<object[]>
            {
                new object[] {3, new[] {0.27901f, 0.44198f, 0.27901f}},
                new object[] {5, new[] {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f}},
                new object[] {7, new[] {0.00598f, 0.060626f, 0.241843f, 0.383103f, 0.241843f, 0.060626f, 0.00598f}},
                new object[]
                {
                    9, new[] {0.000229f, 0.005977f, 0.060598f, 0.241732f, 0.382928f, 0.241732f, 0.060598f, 0.005977f, 0.000229f}
                },
            };
    }
}