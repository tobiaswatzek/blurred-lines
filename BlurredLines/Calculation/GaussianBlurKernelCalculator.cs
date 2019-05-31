using System;
using System.Collections.Generic;
using System.Linq;

namespace BlurredLines.Calculation
{
    /// <summary>
    ///     Calculates gaussian blur kernels.
    ///     The implementation was inspired by http://dev.theomader.com/gaussian-kernel-calculator/. 
    /// </summary>
    public class GaussianBlurKernelCalculator
    {
        /// <summary>
        ///     Calculate a one dimensional kernel for the given <see cref="kernelSize"/>.
        /// </summary>
        /// <param name="kernelSize">An odd number greater than 2.</param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        /// <returns></returns>
        public float[] CalculateOneDimensionalKernel(int kernelSize)
        {
            if (kernelSize < 3 || kernelSize % 2 == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(kernelSize), "Has to be an odd number greater than 2.");
            }


            const double sigma = 1d;
            const double sampleCount = 1000d;


            var samplesPerBin = (int) Math.Ceiling(sampleCount / kernelSize);
            // need an even number of intervals for simpson integration => odd number of samples
            if ((samplesPerBin % 2) == 0)
            {
                samplesPerBin += 1;
            }

            var weightSum = 0d;
            var kernelLeft = -Math.Floor(kernelSize / 2d);

            IList<(double x, double y)> CalcSamplesForRange(double minInclusive,
                double maxInclusive) =>
                SampleInterval((x) => GaussianDistribution(x,
                        0,
                        sigma),
                    minInclusive,
                    maxInclusive,
                    samplesPerBin);

            // get samples left and right of kernel support first
            var outsideSamplesLeft = CalcSamplesForRange(-5 * sigma, kernelLeft - 0.5);
            var outsideSamplesRight = CalcSamplesForRange(-kernelLeft + 0.5, 5 * sigma);

            var allSamples = new List<(IEnumerable<(double x, double y)> samples, double weight)>();
            allSamples.Add((outsideSamplesLeft, 0));

            // now sample kernel taps and calculate tap weights
            for (var tap = 0; tap < kernelSize; ++tap)
            {
                var left = kernelLeft - 0.5 + tap;

                var tapSamples = CalcSamplesForRange(left, left + 1);
                var tapWeight = IntegrateSimpson(tapSamples);

                allSamples.Add((tapSamples, tapWeight));
                weightSum += tapWeight;
            }


            allSamples.Add((outsideSamplesRight, 0));

            // renormalize kernel and round to 6 decimals

            allSamples = allSamples.Select(sample =>
                {
                    sample.weight = Math.Round(sample.weight / weightSum, 6);
                    return sample;
                })
                .ToList();


            return allSamples.Select(sample => (float)Math.Round(sample.weight, 6))
                .Take(allSamples.Count - 1)
                .Skip(1)
                .ToArray();
        }


        private double GaussianDistribution(double x, double mu, double sigma)
        {
            var d = x - mu;
            var n = 1 / (Math.Sqrt(2 * Math.PI) * sigma);
            return Math.Exp(-d * d / (2 * sigma * sigma)) * n;
        }

        private IList<(double x, double y)> SampleInterval(Func<double, double> f,
            double minInclusive,
            double maxInclusive,
            int sampleCount)
        {
            var result = new List<(double x, double y)>();
            var stepSize = (maxInclusive - minInclusive) / (sampleCount - 1);

            for (var s = 0; s < sampleCount; ++s)
            {
                var x = minInclusive + s * stepSize;
                var y = f(x);

                result.Add((x, y));
            }

            return result;
        }

        private double IntegrateSimpson(IList<(double x, double y)> samples)
        {
            var result = samples.First().y + samples.Last().y;

            for (var s = 1; s < samples.Count - 1; ++s)
            {
                var sampleWeight = (s % 2 == 0) ? 2.0 : 4.0;
                result += sampleWeight * samples[s].y;
            }

            var h = (samples.Last().x - samples.First().x) / (samples.Count - 1);
            return result * h / 3.0;
        }
    }
}
