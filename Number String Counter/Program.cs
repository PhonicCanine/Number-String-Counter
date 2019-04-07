using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

using ILGPU.Lightning;
using ILGPU.Lightning.Sequencers;
using ILGPU.Runtime.CPU;

namespace Number_String_Counter
{
    class Program
    {
        //Note: This program runs a *lot* faster in Release mode than Debug mode (because bounds checking is disabled in ILGPU).
        static void Main(string[] args)
        {
            //needed for the other method (Parallel.For)
            //int maxLength = 0;
            //long iterations = 0;

            long originalMin = 0;

            //As a note, it takes around 2 hours on CPU (Core i7 4790K) to search ~120B numbers
            long min = 0;
            long max = 113373373373;

            //Length of an array of longs.
            //Since longs are int64, multiply by ~8 for actual memory use in bytes
            const long allocatedMemory = 200000000;

            //Cache to store results that work in. As this grows, so does the time to search it for good results.
            const int resultCacheSize = 100;

            //Stores the minimum value for the specific depth
            long minForMax = long.MaxValue;

            //The chain length to search for:
            //E.G. 100 (OneHundred) -> 10 (Ten) -> 3 (Three) -> 5 (Five) -> 4 [end] is of length 4.
            const int chainLength = 8;

            //Include punctuation in the count or not.
            //E.G., With punctuation 137 -> "One Hundred and Thirty-Seven" (28 characters)
            //& Without punctuation, 137 -> "OneHundredandThirtySeven" (24 characters)
            const bool includePunctuation = false;

            //Stop when one number is found with specified chain length (obviously invalidates the percentage count)
            const bool stopAtOneFound = false;

            //min is increased as the program runs, so we need a copy of its original value for calculating the % done.
            originalMin = min;

            using (var context = new Context())
            {
                Accelerator acc;

                try
                {
                    acc = new CudaAccelerator(context);
                }
                catch (Exception)
                {
                    //no cuda
                    acc = new CPUAccelerator(context);
                }

                var a = acc;
                Console.WriteLine("Performing ops on " + a.Name + ". " + a.NumMultiprocessors.ToString() + " processors.");

                //Set up two kernels to get the data
                var searchKernel = a.LoadAutoGroupedStreamKernel<Index, ArrayView<UInt64>, long, long, bool>(SearchForChain);
                var resultKernel = a.LoadAutoGroupedStreamKernel<Index, ArrayView<UInt64>, ArrayView<UInt64>>(FindNonZero);

                using (var buffa = a.Allocate<UInt64>((int)allocatedMemory))
                {
                    using (var buffb = a.Allocate<UInt64>(resultCacheSize))
                    {
                        //Loop while we haven't gone over the maximum value in search range
                        while (min < max)
                        {
                            //Search for numbers first (Kernel)
                            searchKernel((int)allocatedMemory, buffa.View, min, chainLength, !includePunctuation);
                            a.Synchronize();

                            //Read back array to find nonzero entries (Kernel)
                            resultKernel(resultCacheSize, buffb, buffa);
                            a.Synchronize();

                            var arr = buffb.GetAsArray();
                            bool found = false;

                            //Read back the results array for nonzero entries (Normal .net)
                            for (int i = 0; i < buffb.Length; i++)
                            {
                                if (arr[i] != 0)
                                {
                                    found = true;
                                    Console.WriteLine(arr[i]);
                                    if (arr[i] < (ulong)minForMax)
                                        minForMax = (long)arr[i];
                                }
                            }

                            //break if we have found a number and had to stop at one found
                            if (found && stopAtOneFound)
                                break;
                            min += allocatedMemory;
                            long total = max - originalMin;
                            long diff = min - originalMin;

                            //For displaying the percentage complete
                            Console.WriteLine((((decimal)diff / (decimal)total)*100).ToString() + "% complete");

                        }
                    }
                }
            }
            

            //The code commented below is for doing this via a Parallel.for.
            /*
            Parallel.For(min,max,(i)=> {
                int chainLength = searchGPU(i,true);
                if (chainLength > maxLength)
                {
                    maxLength = chainLength;
                    Console.WriteLine(NumberToString(i) + "   <=>   (" + i.ToString() + ") gave chain length: " + chainLength.ToString());
                    minForMax = long.MaxValue;
                }

                if (chainLength == maxLength && Math.Abs(i) < Math.Abs(minForMax))
                {
                    minForMax = i;
                    Console.WriteLine(i.ToString() + " was a better candidate for " + chainLength.ToString());
                }

                iterations++;
                if (iterations % ((max - min) / 10000) == 0)
                {

                    decimal percent = (decimal)iterations / (decimal)((max - min));

                    Console.WriteLine((percent * 100)+" Percent done.");
                }
            });*/
            
            Console.WriteLine(NumberToString(minForMax) + "   <=>   (" + minForMax.ToString() + ") gave chain length: " + chainLength.ToString());
        }

        static void FindNonZero(Index index, ArrayView<ulong> needles, ArrayView<ulong> haystack)
        {
            int x = 0;
            for (int i = 0; i < haystack.Length; i++)
            {
                if (haystack[i] != 0)
                {
                    needles[x] = haystack[i];
                    x++;
                }
                if (x >= needles.Length)
                    return;
            }
            for (; x < needles.Length; x++)
            {
                needles[x] = 0;
            }
        }

        //lowest (8) w/  punctuation    323373373
        //lowest (8) w/o punctuation 113373373373
        //Similar to "DoSearchForChain", but entire call tree contains only primative types, so can be compiled for GPU
        static void SearchForChain(Index index, ArrayView<ulong> dataView, long min, long eq, bool removePunctuation)
        {
            long val = (long)searchGPU((long)(index + min),removePunctuation) == eq ? (index + min) : 0;

            if (val > 0)
            {
                dataView[index] = (ulong)val;
            }
            else
            {
                dataView[index] = 0;
            }
        }

        //Non recursive implementation of DoSearchForChain (As recursive functions cannot be compiled for GPU)
        static int searchGPU(long number, bool removePunctuation = true)
        {
            int num = NumberToLength(number, !removePunctuation);
            int chainLength = 1;

            //Keep decomposing the number until we reach 4 (which is always the stopping point in the English language).
            while (num != 4)
            {
                num = NumberToLength(num, !removePunctuation);
                chainLength++;
            }

            //If the number was 4, the chain length is 1. But if the number was 5 (which has four letters, and so will skip the while loop), the chain length should be 2.
            if (number != 4)
                chainLength += 1;

            return chainLength;
        }

        static int DoSearchForChain(long startingNumber, int chainLength = 0, List<long> numbersInChain = null, bool removePunctuation = false)
        {
            string num = NumberToString(startingNumber);
            string puncRemoved = num.Replace(" ", "").Replace(",", "").Replace("-","");
            long nextNumber = (removePunctuation)?(puncRemoved).Length:num.Length;
            if (numbersInChain != null && numbersInChain.Contains(nextNumber))
            {
                return numbersInChain.IndexOf(nextNumber) + 1;
            }
            else
            {
                if (numbersInChain == null)
                {
                    numbersInChain = new List<long>() { startingNumber };
                }

                numbersInChain.Add(nextNumber);

                return DoSearchForChain(nextNumber, chainLength++, numbersInChain, removePunctuation);
            }
        }

        static int NumberToLength(long number, bool punctuationIncluded = true)
        {
            int toReturn = 0;
            int PowerOfThousand = 0;
            if (number == 0)
            {
                return 4;
            }
            while (number > 0)
            {
                long num = number % 1000;
                if (num != 0)
                    toReturn = LessThanOneThousandToInt(num,punctuationIncluded) + (punctuationIncluded ? 1 : 0) + powerThousandCounts(PowerOfThousand) + ((toReturn != 0 && punctuationIncluded) ? 1 : 0) + (punctuationIncluded ? 1 : 0) + toReturn;

                number /= 1000;
                PowerOfThousand++;
            }
            return toReturn - (punctuationIncluded?2:0);
        }

        //Way to pull this out of the function it would otherwise be in for reading clarity.
        static int powerThousandCounts(int idx)
        {
            switch (idx)
            {
                case 0:
                    return 0;
                case 1:
                    return 8;
                case 2:
                    return 7;
                case 3:
                    return 7;
                case 4:
                    return 8;
                case 5:
                    return 11;
                case 6:
                    return 11;
                case 7:
                    return 10;
                case 8:
                    return 10;
                default:
                    return 0;
            }
        }

        static int LessThanOneThousandToInt(long number, bool punctuationIncluded = true)
        {
            int toReturn = 0;
            if (number < 1000)
            {
                long hundreds = (number / 100);
                long tens = (number / 10) % 10;
                long ones = number - (100 * hundreds) - (10 * tens);
                if (hundreds > 0)
                {
                    toReturn += baseNamesLength((int)hundreds) + (punctuationIncluded?8:7) + ((tens == 0 && ones == 0) ? 0 : (punctuationIncluded?5:3));
                }
                if (tens == 1)
                {
                    toReturn += teenNameLengths((int)ones);
                }
                else if (tens == 0)
                {
                    toReturn += baseNamesLength((int)ones);
                }
                else
                {                                                
                    toReturn += tenMultipleNameLengths((int)tens) + (ones != 0 ? (punctuationIncluded?1:0) + baseNamesLength((int)ones) : 0);
                }
                return toReturn;
            }
            else
            {
                return -99999;
            }
        }

        //Same as powerThousandCounts
        static int baseNamesLength(int idx)
        {
            switch (idx)
            {
                case 0:
                    return 0;
                case 1:
                    return 3;
                case 2:
                    return 3;
                case 3:
                    return 5;
                case 4:
                    return 4;
                case 5:
                    return 4;
                case 6:
                    return 3;
                case 7:
                    return 5;
                case 8:
                    return 5;
                case 9:
                    return 4;
                default:
                    return 0;
            }
        }

        //Same as powerThousandCounts
        static int teenNameLengths(int idx)
        {
            switch (idx)
            {
                case 0:
                    return 3;
                case 1:
                    return 6;
                case 2:
                    return 6;
                case 3:
                    return 8;
                case 4:
                    return 8;
                case 5:
                    return 7;
                case 6:
                    return 7;
                case 7:
                    return 9;
                case 8:
                    return 8;
                case 9:
                    return 8;
                default:
                    return 0;
            }
        }

        //Same as powerThousandCounts
        static int tenMultipleNameLengths(int idx)
        {
            switch (idx)
            {
                case 0:
                    return 0;
                case 1:
                    return 0;
                case 2:
                    return 6;
                case 3:
                    return 6;
                case 4:
                    return 6;
                case 5:
                    return 5;
                case 6:
                    return 5;
                case 7:
                    return 7;
                case 8:
                    return 6;
                case 9:
                    return 6;
                default:
                    return 0;
            }
        }

        //Slow, object oriented code that can do the conversion
        static string NumberToString(long number)
        {
            string prepend = "";
            if (number < 0)
            {
                prepend += "Negative ";
                number = number * -1;
            }

            string toReturn = "";

            List<string> PowerOfThousandNames = new List<string>() { "", "Thousand", "Million", "Billion", "Trillion", "Quadrillion", "Quintillion", "Sextillion", "Septillion" };
            int PowerOfThousand = 0;

            while (number > 0)
            {
                long num = number % 1000;
                if (num != 0)
                    toReturn = LessThanOneThousandToString(num) + " " + PowerOfThousandNames[PowerOfThousand] + ((toReturn != "")?",":"") + " " + toReturn;

                number /= 1000;
                PowerOfThousand++;
            }

            toReturn = prepend + toReturn;

            return toReturn.Trim();

        }

        static string LessThanOneThousandToString(long number)
        {
            string toReturn = "";
            List<string> names = new List<string>() { "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine" };
            List<string> teens = new List<string>() { "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen" };
            List<string> tenMultiples = new List<string>() { "", "", "Twenty", "Thirty", "Fourty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };

            if (number < 1000)
            {
                long hundreds = (number / 100);
                long tens = (number / 10) % 10;
                long ones = number - (100 * hundreds) - (10 * tens);
                if (hundreds > 0)
                {
                    toReturn += names[(int)hundreds] + " Hundred" + ((tens == 0 && ones == 0)?"":" and ");
                }
                if (tens == 1)
                {
                    toReturn += teens[(int)ones];
                }else if (tens == 0)
                {
                    toReturn += names[(int)ones];
                }
                else
                {
                    toReturn += tenMultiples[(int)tens] + (ones != 0?"-" + names[(int)ones]:"");
                }
                return toReturn;
            }
            else
            {
                throw new Exception("Yah nah");
            }
        }
    }
}
