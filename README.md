# Number String Counter
Basic .Net core application that counts the length of Number-String chains, utilising the GPU to do so (with ILGPU).

eg.,
```
113373373373 (One Hundred and Thirteen Billion, Three-Hundred and Seventy-Three Million, Three-Hundred and Seventy-Three Thousand, Three-Hundred and Seventy Three) [has 124 non-punctuation characters]
124 (One Hundred and Twenty-Four) [has 23 characters]
23 (Twenty-Three) [has 11 characters]
11 (Eleven) [has 6 characters]
6 (Six) [has 3 characters]
3 (Three) [has 5 characters]
5 (Five) [has 4 characters]
4 (Four) [We end here because 4==("Four").length]
```
is an 8 element long chain.