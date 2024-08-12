## Striver_SDE sheet solution

### 1. Set matrix zero

<details>
	
> You can create a lookup row and col for each cell, the idea is to suppose we are allowed to take extra space and then create row_lookup[c] and col_lookup[r], size is shown in the [], now just iterate and wherever you find 1 mark corresponding position in both lookup tables, finally reiterate the whole matrix and see if you have marked in any of the lookup tables from the both if yes set that cell to 1.

> However we try to save the space by utilizing the 0th r/c of my matrix as a lookup table but to explicitly deal with the mat[0][0] (as it overlaps in both) we have denoted it with the help of the row/col variable.

```cpp
class Solution
{   
    public:
    void booleanMatrix(vector<vector<int> > &mat)
    {
        int r = mat.size();
        int c = mat[0].size();
        int col = 0; 
        int row = 0;
        for(int i = 0 ; i < r ; i++){
            for(int j =0 ; j< c ; j++){
                if(mat[i][j]){
                    if(!i || !j) {
                    if(!j) col = 1;
                    if(!i) row = 1;
                    }
                    else {mat[i][0] = 1 ;mat[0][j] = 1;}
                }
            }
        }
        for(int i = 1 ; i < r ; i++) 
        for(int j = 1 ; j< c ; j++) 
        if(!mat[i][j] && (mat[0][j] || mat[i][0]))
        mat[i][j] = !mat[i][j];
   
        if(row) for(int i = 0 ; i < c ; i++) mat[0][i] = 1;
        if(col) for(int i = 0 ; i < r ; i++) mat[i][0] = 1;
        
           
    }
};
```
</details>

### 2. Pascal Triangle

<details>
	
> n space problem (easy)

```cpp
class Solution {
public:
    vector<vector<int>> generate(int n) {
        vector<vector<int>> an  ;
        vector <int > t ;
        t.push_back(1);
        an.push_back(t);
        n--;
        if(n){
            n--;
            t.push_back(1);
             an.push_back(t);
        }
        while(n--){
            int sz = t.size();
            vector <int> k ;
            k.assign ( t.begin() , t.end());
            t.clear();
            t.assign( 1 + sz  , 0 );
            t[0] = 1 ;
            t[sz] = 1 ;
            int id = 1 ;
            int time = sz - 1;
            while(time--){
                t[id] = k[id-1] + k [id] ; 
                id++;
            }   
            an.push_back(t);
        }
        return an ;
    }
};
```

</details>


### 3. Next Permutation

<details>
	
> The idea is based upon the fact that always remember that when we want to create the next permutation we try to choose the next value as close as possible to the given value, (basically a next greater permutation), ex: 1 3 4 2

> So when you are at some index, and you want to increase it, then ask yourself the numbers lying after it forms the largest number? if yes you can increment the index to the closest possible value that is lying on the right-hand side of it and choose the next possible value otherwise move on from that index towards the right.

> Ex: 1 3 5 4 2: you should move on from 0th index -> you should increment the 1st index since 542 forms the largest possible number among {2, 4, 5} -> 1 4: 5 3 2 ->  L: R (reverse R to get the final number) -> 1 4 2 3 5

> see the code its little tricky bro !!

```cpp
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        // Step 1: Find the break point: kind of decreasing order patern we are iterating
        int ind = -1; // break point
        for (int i = n - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {
                ind = i;
                break;
            }
        }
        // If break point does not exist:
        if (ind == -1) {
            // Reverse the whole array:
            reverse(nums.begin(), nums.end());
            return;
        }
        // Step 2: Find the next greater element and swap it with nums[ind]: since the series is in decreasing order this is only reason why we can break as soon as the if condition is satisfied 
        for (int i = n - 1; i > ind; i--) {
            if (nums[i] > nums[ind]) {
                swap(nums[i], nums[ind]);
                break;
            }
        }
        // Step 3: Reverse the right half:
        reverse(nums.begin() + ind + 1, nums.end());
    }
};
```

</details>
	
### 4. Kadane Algo: Find the subarray with the largest sum, and return its sum.

<details>
	
> The only thing that you should understand here is we break the subarray as soon as the overall sum becomes **-ve**, the only reason for doing so is that taking some -ve sum won't help us further because if you add it you will just decrease the overall maximizing sum that we need. When every element is negative has been handled explicitly.

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& a) {
        int s = 0;
        int ma = INT_MIN;
        int n = a.size();
        for(int i=0; i< n; i++){
            if( s + a[i] < 0 ){
                s=0;
                continue;
            }else{
                s += a[i];
                ma = max(ma , s); // keep track of maximum sub-array sum 
            }
        }
        if(ma == INT_MIN){ // when every elelemnt is negative `ma` will be -infinity 
            for(int i=0; i< n; i++)ma = max ( ma , a[i]);  
        }
        return ma;
    }
};
```
</details>
	
### 5. Sort array having 0,1,2 (three elements only)

<details>
	
> Just remember we need 3 pointers, `l/m/h` and yes `m` is like an iterator throughout the array while `l` and `h` are useful to mark the boundary for `0` and `1` respectively.

```cpp
void sortArray(vector<int>& arr, int n) {
    int low = 0, mid = 0, high = n - 1; // 3 pointers
    while (mid <= high) {
        if (arr[mid] == 0) {
            swap(arr[low], arr[mid]);
            low++;
            mid++;
        }
        else if (arr[mid] == 1) {
            mid++;
        }
        else {
            swap(arr[mid], arr[high]);
            high--;
        }
    }
}
```

</details>
	
### 6. Stock Buy and Sell:

<details>

> You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

> Make a Graph and observe that u have to choose largest slant line inclined towards right, (left is not possible as we need to choose to buy and then sell), Please note two points can be connected from any one to another to observe the required line which we are looking for,

> While traversing the array if SP - CP turns out to be negative better ignore it (since there is option not to sell the stock If you cannot achieve any profit, return 0) and while you traverse since we want to expand the length of our line, so its better if we go deeper at SP end, this is reason why we have minsofar variable.

```cpp
int maxProfit(vector<int> &arr) {
    int maxPro = 0;
    int n = arr.size();
    int minPrice = INT_MAX;

    for (int i = 0; i < arr.size(); i++) {
        minPrice = min(minPrice, arr[i]);
        maxPro = max(maxPro, arr[i] - minPrice); // (sp - cp) think it as a right inclined slant line over which we are choosing the endpoints and subtracting it
    }
    
    return maxPro;
}
```

</details>

### 7. Rotate the matrix 90 degrees clockwise

<details>

> swap(a[i], a[j]) and then reverse every row.

> observe we did this in loop `j < i` reason is we want to traverse lower triangular matrix only.

```cpp
void rotate(vector < vector < int >> & matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) swap(matrix[i][j], matrix[j][i]);
    for (int i = 0; i < n; i++) reverse(matrix[i].begin(), matrix[i].end());
}
```

</details>

### 8. Merge overlapping subintervals

<details>

> Sort `iv` based on the first element and then the second element, in your hand, pick the first interval and say it is `cur` now with further intervals [1 to iv. size()-1]. Check whether there is overlap, if yes try expanding the cur interval to max boundary points else push it to the answer and choose a new interval as your `cur`.

```cpp
class Solution {
public:
    static bool compare(const vector<int>& a, const vector<int>& b) {
        if (a[0] == b[0])  return a[1] < b[1]; // second element based sorting (ascending)
        return a[0] < b[0]; // first elemtent based sorting (ascending)
    }
    vector<vector<int>> merge(vector<vector<int>>& iv) {
        // Sort iv based on the first element and then the second element
        sort(iv.begin(), iv.end(), compare);
        vector<vector<int>> an;
        vector<int> cur = iv[0];
        for (int i = 1; i < iv.size(); i++) {
            if (cur[1] >= iv[i][0])  cur[1] = max(cur[1], iv[i][1]); // we are extending the length of merged interval as u can see stored in variable `cur`
             else {
                an.push_back(cur);
                cur = iv[i];
            }
        }
        an.push_back(cur); // Don't forget to add the last merged interval
        return an;
    }
};
```

</details>


### 9. Merge 2 sorted array (no extra space)

<details>
	
> To do it in `O(n+m)` you can take the extra space and do it, however, that approach is similar to what we did in merge sort, but to do it in 1 space we need `O(nlgn)` time, now we assume that `a` will keep the smallest element at [0] while `b` keeps the largest element at last in it, so please remember the initialization which has utmost importance = Place the pointer at the last of `a` and at the beginning of the array `b`. A way to remember is take two array so when you join them, pointers are supposed to be placed at the joints.

```cpp
void mergeTwoSortedArraysWithoutExtraSpace(vector<long long> &a, vector<long long> &b){
	int n = a.size();
	int m = b.size();
	int i =  n - 1 ;
	int j = 0 ;
	while(i>=0 && j < m){
          if (a[i] > b[j]) {
          swap(a[i], b[j]);
		  i--;
		  j++;
		  }else break;
        }
	sort(a.begin(),a.end());
	sort(b.begin(),b.end());
	
}
```

</details>
	
### 10. Find duplicate in array of N+1 integers

<details>
	
> Refer Q.11 M3

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        for(int i = 0 ; i < nums.size() ; i++) {
            int id = abs(nums[i]);
            if(nums[id]>0) nums[id]*=-1;
            else return id;
        }
        return 0; // dummy
    }
};
```

</details>


### 11. Repeat and missing number

<details>
	
> M1: Let the array size be n, so as per the expectations 1 to n numbers were supposed to be present in it. However, one guy is missing while the other guy is being repeated. We use BF to find the missing/repeating number by having doubt over every single number from 1 to n  and then checking it with the given array. Clearly, it takes `O(2*n^2)`.

> M2: Use the hash map, say 1 to n is my index of an array whenever we find a number we do ++ over there, clearly when you reiterate the hash array and if you find somewhere 0 b/w 1 to n that means that index is my missing number, a corollary to that is if you find somewhere as 2 stored that means that index is being repeated.

> M3: Iterate the array and just imagine that array is 1 indexed, now your job is that for every index `i`, go to a[i]th index and mark the value stored with a `-ve` sign, however, if it is already marked that means a[i] is my repeating number, in that case don't mark it, and kindly continue the process, at the end you will see that one of the ith index is still having a `+ve` sign that index is actually the missing one.  

> M4: Maths, say the sum of the array is s1 while the sum of squares of elements is s2, say y is your repeating number while x is the missing, we derive a formula as `x-y = n(n+1)/2 -s1` and `x^2 - y^2 = n(n+1)(2n+1)/6 - s2`, clearly you have x-y and x+y so congratulations!!

> M5: This is kind of unobvious, an XOR trick, so take the xor of all array elements and then keep on taking the xor till 1 to n now at the end you will be left with a xor value which actually represents x^y, now we need to segregate the x and y from x^y which is very tricky, please see the code, and understand yourself.

```cpp
vector<int> findMissingRepeatingNumbers(vector<int> a) {
    int n = a.size(); 
    int xr = 0;
    //Step 1: Find XOR of all elements:
    for (int i = 0; i < n; i++) {
        xr = xr ^ a[i];
        xr = xr ^ (i + 1);
    }
    //Step 2: Find the differentiating bit number: right most significant bit, reason is those two numbers (x, y) will actually differ at this bit only
    int number = (xr & ~(xr - 1));
    //Step 3: Group the numbers:
    int zero = 0;
    int one = 0;
    for (int i = 0; i < n; i++) {
        //part of 1 group:
        if ((a[i] & number) != 0) {
            one = one ^ a[i];
        }
        //part of 0 group:
        else {
            zero = zero ^ a[i];
        }
    }

    for (int i = 1; i <= n; i++) {
        //part of 1 group:
        if ((i & number) != 0) {
            one = one ^ i;
        }
        //part of 0 group:
        else {
            zero = zero ^ i;
        }
    }

    // Last step: Identify the numbers: so one and zero are basically (x, y) after u have done group segregation with 1 - n and all array numbers 
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] == zero) cnt++;
    }

    if (cnt == 2) return {zero, one};
    return {one, zero};
}
```

### 11.1. Extension problem: Imagine if you had elements from 0 to n-1 and an array of size n, then let's say elements, appear more than once like n=10: [0 1 0 0 1 1 5 9 9 0], how do we find all duplicates and missing exactly once in `n` time and `1` space?  Here what we do is we go to arr[i] index, and increase that index by n every time, the rest code is self-explanatory.

```cpp
class Solution {
public:
    vector<int> duplicates(int arr[], int n)
{
    vector<int> occured_twice_or_more, missing, occured_just_once;
    for (int i = 0; i < n; i++) {
        int index = arr[i] % n;
        arr[index] += n;
    }
    for (int i = 0; i < n; i++) 
        {
           if ((arr[i] / n) >= 2) occured_twice_or_more.push_back(i);
           if(arr[i] < n) missing.push_back(i);
           if ((arr[i] / n) == 1) occured_just_once.push_back(i);
        }
  for (int i = 0 ; i< missing.size(); i++) cout<<missing[i]<<" ";
  cout<<endl;
  for (int i = 0 ; i< occured_just_once.size(); i++) cout<<occured_just_once[i]<<" ";
  cout<<endl;
  if(occured_twice_or_more.size()) return occured_twice_or_more;
  else return {-1};
}
};
```

</details>

### 12. Inversion of array 

<details>

> [explanation1](https://www.youtube.com/watch?v=AseUmwVNaoY&t=364s) [mergesort-vid-animation](https://www.youtube.com/watch?v=5Z9dn2WTg9o)

> You should note that we recur for the left and right parts so that also gives me an inversion count we need to add both and when you merge the two halves at any step it also generates the count so we need to add up the count for three cases, now when we go to merge and we say while merging if a guy from left half becomes greater than any guy on the right then everyone following the guy on the left side including it will be greater than the right half guy, so that's what is used to count the inversions. `cnt += (mid - left + 1); //Trick`

> low ______ left _________ mid     mid+1 _______ right ________ high, so if `a[left] > a[right]` then we are sure, left to mid `(mid - left + 1)` are also greater than `a[right]`, this forms the intuition for this problem 

```cpp
int merge(vector<int> &arr, int low, int mid, int high) {
    vector<int> temp;
    int left = low;     
    int right = mid + 1;  
    int cnt = 0;
    while (left <= mid && right <= high) {
        if (arr[left] <= arr[right]) {
            temp.push_back(arr[left]);
            left++;
        }
        else {
            temp.push_back(arr[right]);
            cnt += (mid - left + 1); //Trick
            right++;
        }
    }
    while (left <= mid) {
        temp.push_back(arr[left]);
        left++;
    }
    while (right <= high) {
        temp.push_back(arr[right]);
        right++;
    }
    for (int i = low; i <= high; i++)  arr[i] = temp[i - low];
    return cnt; 
}

int mergeSort(vector<int> &arr, int low, int high) {
    int cnt = 0;
    if (low >= high) return cnt;
    int mid = (low + high) / 2 ;
    cnt += mergeSort(arr, low, mid);  // left half count
    cnt += mergeSort(arr, mid + 1, high); // right half count
    cnt += merge(arr, low, mid, high);  // merging sorted halves count
    return cnt;
}

int numberOfInversions(vector<int>&a, int n) {
    return mergeSort(a, 0, n - 1);
}
```

</details>

 
### 13. Search in 2D matrix

<details>
	
> You can actaully flatten `2D` matrix to `1D` array and you will have sorted array tbh, that's what we have been doing but cleverly 

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int n = matrix.size();
    int m = matrix[0].size();
    int low = 0, high = n * m - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        int row = mid / m, col = mid % m;
        if (matrix[row][col] == target) return true;
        else if (matrix[row][col] < target) low = mid + 1;
        else high = mid - 1;
    }
    return false;
}
```

</details>
	
### 14. Pow(x,n) x^n 

<details>
	
```cpp
class Solution {
public:
    double myPow(double x, int n) {
        if (n == 0)   return 1.0;
        double result = 1.0;
        long long absN = abs(static_cast<long long>(n));
        while (absN > 0) {
            if (absN & 1)  result *= x;
            x *= x;
            absN >>= 1;
        }
        return (n < 0) ? (1.0 / result) : result;
    }
};
```

</details>

### 15. Majority (>n/2)

<details>

```cpp
class Solution {
public:
    int majorityElement(vector<int>& a) {
        int f1 ; // to store a candidate element 
        int c1 = 0; // to count the occurences of candidate element
        for (auto no: a){
            if(no == f1) c1++;
            else if (c1 == 0) {f1 = no; c1 = 1;}
            else c1--;
        }
        int cnt = 0 ;
        for (auto no: a) if(f1 == no) cnt++;
        if(cnt > a.size()/2) return f1;
        return -1;
    }
};
```

</details>
	
### 16. Majority (>n/3)

<details>

```cpp
class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        int f1 = 0, f2 = 0; // to store a candidate elements
        int c1 = 0, c2 = 0; // to count the occurences of candidate elements
        for (int num : nums) {
            if (num == f1) c1++;
            else if (num == f2) c2++;
            else if (c1 == 0) {
                f1 = num;
                c1 = 1;
            } else if (c2 == 0) {
                f2 = num;
                c2 = 1;
            } else {
                c1--;
                c2--;
            }
        }
        c1 = c2 = 0;
        for (int num : nums) {
            if (num == f1) {
                c1++;
            } else if (num == f2) {
                c2++;
            }
        }
        vector<int> result;
        if (c1 > nums.size() / 3) {
            result.push_back(f1);
        }
        if (c2 > nums.size() / 3) {
            result.push_back(f2);
        }
        return result;
    }
};
```

</details>

### 17. Grid unique paths

<details>
	
```cpp
class Solution {
public:
     int countPaths(int i,int j,int n,int m)
    {
        if(i==(n-1)&&j==(m-1)) return 1;
        if(i>=n||j>=m) return 0;
        else return countPaths(i+1,j,n,m)+countPaths(i,j+1,n,m);
    }
    int uniquePaths(int m, int n) {
       return countPaths(0,0,m,n);
    }
};
```
</details>
	
### 18. Reverse pairs

<details>

> Imagine two sorted array and we need to count such that `i < j && a[i] > 2a[j]` so imagine `i` is on left sorted array and `j` is on the right sorted array so we are done with `i < j`, now 
	
> Refer [Q.12](https://github.com/kuspia/Striver_SDE-shortnotes-/tree/main#12-inversion-of-array-explanation1-mergesort-vid-animation)

> left(low) ______ i _________ mid     mid+1(right) _______ j ________ high, so if `a[i] > 2 * a[j]` then just a observation `right` to `j` will also satisfy the condition and that's what form the intution of the problem, notice we increment `i` and `j` in not more than `n` cost, we did it cleverly

```cpp
class Solution {
public:
int merge(vector<int> &arr, int low, int mid, int high) {
    vector<int> temp;
    int left = low;     
    int right = mid + 1;  
    long long cnt = 0; 
    int j = mid +1 ;
    for (int i = low; i <= mid; ++i) { while (j <= high && arr[i] > 2LL * arr[j]) ++j;
    cnt += (j - (mid + 1)); // notice j is already (j++) that's why we have done this way
    }

    while (left <= mid && right <= high) {
        if (arr[left] <= arr[right]) {
            temp.push_back(arr[left]);
            left++;
        }
        else {
            temp.push_back(arr[right]);
            //if(arr[left] > 2LL * arr[right]) cnt += (mid - left + 1); // wrong won't work 
            right++;
        }
    }
    while (left <= mid) {
        temp.push_back(arr[left]);
        left++;
    }
    while (right <= high) {
        temp.push_back(arr[right]);
        right++;
    }
    for (int i = low; i <= high; i++)  arr[i] = temp[i - low];
    return cnt; 
}
int mergeSort(vector<int> &arr, int low, int high) {
    if (low >= high) return 0; // array has just one element or you are accessing something invalid
    long long cnt = 0; 
    int mid = (low + high) / 2 ;
    cnt += mergeSort(arr, low, mid);  // left half count
    cnt += mergeSort(arr, mid + 1, high); // right half count
    cnt += merge(arr, low, mid, high);  // merging sorted halves count
    return cnt;
}
int numberOfInversions(vector<int>&a, int n) {
    return mergeSort(a, 0, n - 1);
}
int reversePairs(vector<int>& a) {
    return numberOfInversions(a, a.size());
}
```
</details>

### 19. 2 sum 

<details>

> BF ways are like sort and then do BS, or maybe use hashing etc.

> Refer 4 sum 

```cpp
```

</details>


### 20. 4 sum

<details>

> The optimized way is `O(n^3)`, so choose two elements using the `O(n^2)` loop, while the other two can be picked up using **two pointers** approach, Please note to skip duplicates for `i`, `j`, `p`, `q` pointers.
	
```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& a, int k) {
        vector<vector<int>> an;
        sort(a.begin(), a.end()); // see this bro -< something that let you use two pointers approach 
        int n = a.size();
        for (int i = 0; i < n ; i++) {
            if (i > 0 && a[i] == a[i - 1]) continue; 
            for (int j = i + 1; j < n; j++) {
                if (j > i + 1 && a[j] == a[j - 1]) continue; 
                int p = j + 1;
                int q = n - 1;
                while (q > p) {
                    int sum = a[i] + a[j] + a[p] + a[q];
                    if (sum == k) {
                        vector<int> tmp = {a[i], a[j], a[p], a[q]};
                        an.push_back(tmp);
                        while (p < q && a[p] == tmp[2]) p++; 
                        while (p < q && a[q] == tmp[3]) q--; 
                    } else if (sum > k) {
                        q--;
                    } else {
                        p++;
                    }
                }
            }
        }
        return an;
    }
};
```

</details>


### 21. Longest Consecutive Subsequence

<details>

> Use set (ordered, hence sorted) to store the numbers to avoid duplicates, now simply try maintaining the length of the longest sequence encountered so far which is consecutive in nature.

> Example: 1 8 2 1 1 3 5 5 6 7 7-> set(1 2 3 5 6 7 8) -> 1 2 3 or 5 6 7 8 -> length max: 4 answer
	
```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& n) {
        set<int> s;
        for(int i = 0; i < n.size(); i++) s.insert(n[i]);
        int longestLen = 0; // longest consecutive sequence length
        int currentLen = 0; // current consecutive sequence length
        int prev = INT_MIN; // previous element

        for (int num : s) {
            if (num == prev + 1 ) currentLen++;
            else  currentLen = 1;
            longestLen = max(longestLen, currentLen);
            prev = num;
        }
        return longestLen;
    }
};
```

> Linear time solution: The time complexity is computed under the assumption that we are using unordered_set and it is taking `O(1)` for the set operations. Time Complexity: `O(N) + O(2*N) ~ O(3*N)`, where `N` = size of the array.

> Let's understand time complexity: consider this case: [1] 2 3 [6] 7 8 [10] 11 12 13: inside the set (may be in any order), now the first outer loop runs for `n` time, that's fine, now thing is we are picking only starting elements i.e. `1`, `6`, `10`, now when we pick `1` we iterate only till consecutive length which is equals to `3`, cleary we are clear we do `n` times iterations only inside the second loop.

> [Youtube](https://www.youtube.com/watch?v=oO5uLE7EUlM&t=865s)

```cpp
int longestSuccessiveElements(vector<int>&a) {
    int n = a.size();
    if (n == 0) return 0; // we are working with non-zero vector 
    int longest = 1;
    unordered_set<int> st;
   for (int i = 0; i < n; i++)  st.insert(a[i]);
    for (auto it : st) {
        //if 'it' is a starting number:
        if (st.find(it - 1) == st.end()) { // very very important optimization !!
            //find consecutive numbers:
            int cnt = 1;
            int x = it;
            while (st.find(x + 1) != st.end()) {
                x = x + 1;
                cnt = cnt + 1;
            }
            longest = max(longest, cnt);
        }
    }
    return longest;
}
```

</details>



### 22. Longest subarray with given sum K

<details>

> To find subarray with sum `k` brute force is to consider all subarrays that cost you n^3 (worst BF approach) however it can be brought down to (n^2) in case you think a little bit, but still it is BF only.

> Join your hands and say with me that I will keep on moving my right hand away from my left hand unless the sum is not equal to k, the moment I hit it as k it's my duty to record the gap, however, if the sum exceeds the given `k` it's the time to move the left hand towards right unless the sum is greater than k, and yeah at every step keep on decreasing the sum with value to which your left hand was pointing.

> Try seeing the code unless you haven't learned it and remember how we did the initial initialization of my variable {left, right, sum}. TC: O(2*N)

```cpp
int longestSubarrayWithSumK(vector<int> a, long long k) {
    int n = a.size(); 
    int left = 0, right = 0; 
    long long sum = a[0];
    int maxLen = 0;
    while (right < n) { // n complexity 
        while (left <= right && sum > k) { // overall this loop runs for n times hence contribute n time complexity throught the code 
            sum -= a[left];
            left++;
        }
        if (sum == k) maxLen = max(maxLen, right - left + 1);
        right++;
        if (right < n) sum += a[right];
    }
    return maxLen;
}
```

> What if my numbers are 0 or negative? There comes a technique that uses the hash map, so the idea is to just store the prefix sum with it's index as the value in the map, remember if you hit the same prefix sum value don't update it because we need to maximize the length of my subarray.
                           
> Idea: _____ (x-k) _______ | _____ k ____ (let is be a array whose sum(prefix) is x at some point, now if I look back and I found a prefix sum = (x-k), this implies that yes we encountered a subarray just now whose sum = k
	
```cpp
int getLongestSubarray(vector<int>& a, long long k) {
    int n = a.size(); // size of the array.
    map<long long, int> preSumMap;
    long long sum = 0;
    int maxLen = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
        if (sum == k) { // Case when your prefix sum is exactly equal to k so just `i+1` will be your answer
            maxLen = max(maxLen, i + 1);
        }
        // calculate the sum of remaining part i.e. x-k:
        long long rem = sum - k;
        if (preSumMap.find(rem) != preSumMap.end()) { // Look for `rem` in our map
            int len = i - preSumMap[rem];
            maxLen = max(maxLen, len);
        }
        //Finally, update the map iff prefix sum is not present in our map
        if (preSumMap.find(sum) == preSumMap.end()) {
            preSumMap[sum] = i;
        }
    }
    return maxLen;
}
```

</details>

### 23. Number of subarrays with xor = b

<details>
	
> ______ x^b ______ ? ________ let till this point xor is `x` Assume that before that we have encountered a xor value = `x^b`, then `?` has to be = `b` only, to satisfy the property `(x^b ^ ? = x) => ? = b` so this is how we got a subarray with xor value = `b`

> ______ x^b ______ b _______ : let you have `x` in your `xorSum`, and we can have `x^b` occuring multiple times previously, since we need to count all of them, so we need to maintain counter, to get all possible sub-arrays with xor = `b` when `xorSum` = `x`	
	
```cpp
int Solution::solve(vector<int> &A, int B) {
    int n = A.size();
    int xorSum = 0;
    int count = 0;
    unordered_map<int, int> xorCount;
    for (int i = 0; i < n; i++) {
        xorSum ^= A[i];
        if (xorSum == B)  count++;
        if (xorCount.find(xorSum ^ B) != xorCount.end())  count += xorCount[xorSum ^ B];
        xorCount[xorSum]++;
    }
    return count;
}
```
</details>

### 24. Longest substring without repeat

<details>
	
> Join two hands and keep them at one distance move your right hand unless you keep on finding the unique characters the moment you see a repeated character it is your responsibility to shift your left hand and keep one distance ahead of the position where the last occurrence of repeated character was found.

> So, imagine we have a string with `size>=2`, now we place `i` at `0` and `j` at `1`, and iterate unless `j` doesn't hit `n`, we say that if `s[j]` was previously found then `i` should be shifted by unity towards the right-hand side, as we always try to maintain the unique characters in my constructed substring from `i` to `j`, therefore if we shift it by one we ensure this property. However, we have to also make sure `i` should be a part of the current substring therefore we use max() very important thing (dry run with `abba` or `abcdbea`) to get the gist.
	
```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        map <char , int > mp ;
        int i = 0 ;
        int j = 1 ;
        int ma = INT_MIN;
        mp[s[0]] = 0 ;
        while (j < s.size() ){
            if( mp.find( s[j]) != mp.end() ) i = max(i, mp[s[j]] + 1);  // often people think to do `mp[s[j]]+1`, only reason to take maximum with `i` is to ensure that we don't cosnider the string before to ith index, or you can say the string which is not even the part of our current string (i to j)  // abba dry run to get the gist 
            mp[s[j]] = j ; // record every char index value
            ma = max (ma , j-i+1); // keep on updating maximum length every time, who knows when we might get longest substring
            j++;
        }
        return ma==INT_MIN ? s.size(): ma; // true condition explicitly checks the string of size 1 or 2
    }
};
```

</details>

### 25. Reverse a LL
<details>
	
> You should remember this as we often require this reversal technique most of the time. nh(NULL) : <- (todo) : [h] -> [] -> [] -> null

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* h) {
        ListNode* nh = NULL;
        while(h){
            ListNode* temp = h->next;
            h->next = nh;
            nh = h;
            h = temp;
        }
        return nh;
    }
};
```
</details>

### 26. Middle of LL

<details>

> Use s/f pointers, easy

```cpp
class Solution {
public:
    ListNode* middleNode(ListNode* h) {
        if(!h->next) return h; // only one node is present
        ListNode *s, *f;
        s = h;
        f = h;
        while(f){
            s = s->next;
            f = f->next;
            f = f->next;
            if(!f || !f->next) return s;
        }
        return NULL; // dummy 
    }
};
```
</details>

### 27. Merge two sorted LL

<details>

> Refer [Q.36](https://github.com/kuspia/Striver_SDE-shortnotes-/blob/main/README.md#36-flattening-of-ll)

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode *dh = new ListNode();
        ListNode *cur = dh;
        while(list1 && list2){
            if(list1->val>list2->val){
                cur->next = list2;
                list2 = list2->next; 
            }else{
                cur->next = list1;
                list1 = list1->next;
            }
            cur = cur->next;
        }
        if(list1) cur ->next = list1;
        if(list2) cur-> next = list2;
        return dh->next;
    }
};
```
</details>

### 28. Remove Nth node from back of LL 

<details>

> Intuition: imagine a line (L): _____________|___ 2cm ____ now if we place another line from start say of 2cm and marks both ends as `s` and `f`, clearly when we drag that line along the (L) and as soon as `f` reaches the end, we can say `s` is pointing to start of 2cm from the end.

> The trick is to create a dummy head and point it to the head, while slow and fast pointers store the address of the dummy head initially, now move fast by n+1 times, and then move slow and fast both unless fast doesn't hit NULL, doing so you will observe slow points to a node just before the nth node from the end.

> [dh]`(s/f)` - - - > [h] -> [] -> [] -> null

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* h, int n) {
        ListNode *s, *f, *dh;
        dh = new ListNode();
        dh->next = h;
        s=dh;
        f=dh;
        n++;
        while(n--) f = f->next;
        while(f){
            s = s->next;
            f = f->next;
        }
        s->next = s->next->next; // reconnect the LL and drop the nth node 
        return dh->next;
    }
};
```
</details>

### 29. Add two numbers as LLs (numbers are stored in reverse order)

<details>
	
> <img width="464" alt="Screenshot 2024-08-11 at 12 04 22" src="https://github.com/user-attachments/assets/94fd8e9c-b38f-4d3f-818b-3b61c960e25b">

> The idea is to create a new `dh`, and point `h` to `dh`, please note that `h` is like a node traversal pointer in our newly created LL that stores the sum block by block, clearly the question is very easy however you should see till when we are running the while loop and how we initialize `s` to `c` then add list values if there exist any.

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *dh = new ListNode ();
        ListNode *h = dh;
        int c = 0;
        while(l1||l2||c){
            int s = c;
            if(l1){
                s+=l1->val;
                l1=l1->next;
            }
            if(l2){
                s+=l2->val;
                l2=l2->next;
            }
            c=s/10;
          ListNode *block = new ListNode (s%10, NULL);
          h->next = block;
          h=block;
        } 
        return dh->next; 
    }
};
```
</details>

### 30. Delete a given node from LL O(1)

<details>

> What we do is we copy the value to our current node of it's next node and then point the current node to the next to next node, finally we release the current node's next node however, it is not mandatory.

```cpp
class Solution {
public:
    void deleteNode(ListNode* n){
        ListNode* nxt = n->next;
        n->val = nxt->val;
        n->next = nxt->next;
        delete nxt; 
    }
};
```

</details>

### 31. Intersection of two LL (Y structure pattern not X pattern)

<details>

> Draw a big `Y` with one line longer than another to get the gist of both intution.

> <img width="269" alt="Screenshot 2024-08-11 at 17 13 34" src="https://github.com/user-attachments/assets/93d68714-59f2-45ef-b1c3-55ebb12a3268">

> A tricky question so what you have to do is find the length of LL and take the absolute difference of both, now if is greater than 0 then please travel longer LL by that many steps, and after that move by one step in both LL to find the meeting point.

> Another way is to travel both LLs by one step and when you reach the null, just exchange the traversal path by pointing the iterator to the other path head node.  

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *head1, ListNode *head2) {
    ListNode* d1 = head1;
    ListNode* d2 = head2;
    
    while(d1 != d2) {
        d1 = d1 == NULL? head2:d1->next; // we assign d1 and at same time we are checking had we arrrived the null 
        d2 = d2 == NULL? head1:d2->next; // we assign d2 and at same time we are checking had we arrrived the null 
    }
    return d1;
    }
};
```

</details>

### 32. Cycle in LL

<details>

> Maintain slow and fast pointers where slow and fast have different speeds of traversing the LL, if at any point you encounter a null pointed by slow or fast that means there is no cycle else those slow and fast pointers will definitely meet that confirms there is a cycle and we return 1.

> Please note that they may meet anywhere at any node in the cycle not necessarily at the starting point or somewhere. Also, see our fast pointer try to catch the slow one, the slow pointer is guaranteed to be caught in one completion round of the circle it can't take more than one round however fast may take `n` complete rotations + a few steps from the starting point of the circle to catch the slow.

> So let's say from head to starting point distance is `l1`, and the point of meetup from the starting point of the circle is at distance `l2` so slow pointer will travel l1+l2 distance while fast will travel `l1+l2+nc` distance => [`2*(l1+l2)` `=` `l1+l2+nc`], clearly fast travel twice the distance slow does, therefore you can get a relation as `l1 = nc -l2`, so if you need to locate the starting point then move a pointer from the head and move another pointer from the meetup point by one step and wherever they will meet will be your starting point of the circle. 

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *s = head ;
        ListNode *f = head ;
        while(1){
            if(!s) return 0;
            s = s-> next ;
            if(!s) return 0;
            f = f-> next ;
            if(!f) return 0;
            f = f->next;
            if(!f) return 0;
            if(s == f) return 1;
        }
    }
};
```

</details>

### 33. Reverse a LL in a group of size k

<details>
	
> <img width="427" alt="Screenshot 2024-08-11 at 17 50 49" src="https://github.com/user-attachments/assets/fc762480-fded-4f0f-9ea7-850c3d427ca6">

> Self-explanatory code, very interesting just see it at least 2-3 times, the idea is to check this: `if (len < k) return h` before you process any recursion call if it fails then simply reverse the `k` size LL from the current state value of `h` using two pointers `prev` and `current` which is very easy and clearly h->next in your current state of recursion call will point to some node that will be returned by the next recursion call which will accept new head as the `current` pointer value. The new head of the reversed group will be pointed by your `prev` pointer if you observe carefully.

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* h, int k) {

        int len = 0;
        ListNode* current = h;
        while (current) {len++;current = current->next;}
        // Check if there are at least k nodes in the remaining part
        if (len < k) return h;
        
        // Reverse the first k nodes in the current group
        ListNode* prev = nullptr;
        current = h;
        for (int i = 0; i < k; i++) {
            ListNode* tmp = current->next;
            current->next = prev;
            prev = current;
            current = tmp;
        }
        
        // Recursively reverse the next k nodes and connect the groups
        h->next = reverseKGroup(current, k);
        
        // Return the new head of the reversed group
        return prev;
    }
};
```

</details>

### 34. Palindrome LL

<details>

> Easy, find the length and then reverse the next half of LL and maintain two pinter one at the beginning and one at the start of the reversed right half of the LL, compare one by one, please note first see the even length code then odd (we have skipped middle element in the case of odd).

```cpp
class Solution {
public:
ListNode* reverse(ListNode* h){
ListNode* prev = NULL;
while(h){
    ListNode* nxt = h->next;
    h->next = prev;
    prev = h;
    h = nxt;
}
return prev;
}
    bool isPalindrome(ListNode* h) {
        if( ! h->next ) return 1; // case of 1 node 
        ListNode *dh, *pt;
        dh = h;
        int len =0;
        while(dh){
            len++;
            dh= dh->next;
        }
        if(len&1){
             dh = h; // 1 2 3 4 5
            int op = len / 2; // op is 2
            while (op--) dh = dh->next; // dh points at 3 now
            dh->next = reverse(dh->next);  // 1 2 3 5 4
            dh = dh->next; // dh points at 5 now (skipped middle element cleverly as the length was odd
            pt = h;
            while (pt->val == dh->val) {
                pt = pt->next;
                dh = dh->next;
                if (!dh) return 1;
            }
        }else{
            dh = h;
            int op = len/2;
            op--;
            while(op--) dh = dh->next ; // 1 2 3 4 5 6 
            dh->next = reverse(dh->next); // here for above example dh will point at // 1 2 3 6 5 4
            dh = dh->next ; // now dh points to 6
            pt = h ;
            while(pt->val == dh->val){
                pt = pt->next;
                dh = dh->next;
                if(!dh) return 1; 
            }
        }
        return 0; 
    }
};
```

</details>

### 35. Starting point of loop in LL

<details>

> Refer [Q.32](https://github.com/kuspia/Striver_SDE-shortnotes-#32-cycle-in-ll)

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *s = head ;
        ListNode *f = head ;
        while(1){
            if(!s) return NULL;
            s = s-> next ;
            if(!s) return NULL;
            f = f-> next ;
            if(!f) return NULL;
            f = f->next;
            if(!f) return NULL;
            if(s == f) break;
        }
        while(head!=s){
            s = s->next;
            head = head->next;
        }
        return s;
    }
};
```

</details>


### 36. Flattening of LL

<details>

> <img width="573" alt="Screenshot 2024-08-11 at 11 27 39" src="https://github.com/user-attachments/assets/1d70d1b9-105b-4715-a3d8-7e32cdca3e4f">

> Good question so FYI LL is sorted from top to bottom and from left to right, we are supposed to flatten it and give the final sorted LL. The thought process is that we store the top node addresses of each LL in the vector and reverse it to process it, I mean as we have a lot of sorted LLs we assume our ans as the last LL and then we merge it from the back side one by one, this is very obvious as we finally need to make the sorted flattened LL. Don't forget to do `ans -> next = NULL` at the end and also make sure the final sorted LL: `ans`  last bottom node also points to NULL. FYI the final LL will be stored in a vertical manner and also every node's next pointer points to NULL in the `ans`.

> So as you are clear with the question now we have to simply merge two sorted LL which is very easy but still give it a dry run to have the gist of how it really works.

> Create a dummy node to simplify the merging process, yes just learn this and remember to make a copy of dummy node k/as current that moves and helps you to merge both of them.

> Why from back? TO-write

```cpp
Node* merge(Node* r1, Node* r2) {
    Node* dummy = new Node(0); 
    Node* current = dummy;
    while (r1 && r2) {
        if (r1->data <= r2->data) {
            current->bottom = r1;
            r1 = r1->bottom;
        } else {
            current->bottom = r2;
            r2 = r2->bottom;
        }
        current = current->bottom;
    }
    // Attach the remaining nodes (if any)
    if (r1)  current->bottom = r1;
    else  current->bottom = r2;
    return dummy->bottom; // The merged list starts from the next of the dummy node
}
Node *flatten(Node *r)
{
   if(! r->next) return r;
   vector<Node*> v;
   Node *d , *ans;
   d=r;
   while(d){
       v.push_back(d);
       d = d -> next;
   }
   reverse (v.begin(), v.end());
   ans = v[0];
   for(int i =1; i <= v.size()-1 ; i++) ans = merge(ans, v[i]);
   ans -> next = NULL;
   return ans;
}
```

</details>

### 37. Rotate a LL

<details>
> <img width="286" alt="Screenshot 2024-08-11 at 18 17 16" src="https://github.com/user-attachments/assets/38f89328-dc0a-4b5e-95c5-23cc5520aca7">

> Please note we need to find the length and then point the tail node to the head and k+1th node to null counted from the right-hand side in the original LL.

```cpp
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (!head || k == 0)  return head;
        int length = 1;
        ListNode* tail = head;
        while (tail->next) {
            length++;
            tail = tail->next;
        }
        k = k % length; // Calculate effective rotation amount
        if (k == 0)   return head; // No rotation needed
        tail->next = head; // Connect the tail to the head to form a loop
        for (int i = 0; i < length - k; i++)  tail = tail->next;
        ListNode* newHead = tail->next;
        tail->next = nullptr;
        return newHead;
    }
};
```

</details>

### 38. Clone a LL with random and next pointer

<details>

> <img width="1118" alt="Screenshot 2024-08-11 at 18 22 53" src="https://github.com/user-attachments/assets/696f57e7-9b14-4c84-b6dc-bfa275d64d0c">

> <img width="1139" alt="Screenshot 2024-08-11 at 18 24 54" src="https://github.com/user-attachments/assets/5bd2bd27-7eb2-4a42-9f02-88556a2eb5a1">


> Idea is based on the fact that we create a new node between every two nodes and connect the next pointers of each other after that we try to connect random pointers and then the next pointers, make sure you preserve the old LL connections too at the end, connecting next pointers is kinda tricky however random is very easy.


```cpp
class Solution {
public:
    Node* copyRandomList(Node* h) {
        if(!h) return h;
        Node* th = h;
        Node* nh = NULL ;
        // trying to create a new node between every two nodes and connecting the next pointers of each other
        while(th){ 
         Node *new_node = new Node(th->val);
            Node* temp = th -> next ;
            th->next = new_node;
            new_node->next = temp;
            th = th -> next->next ;
        }
        th = h;
        while(th){ // connect the new LL random pointers 
            if(th->random) th->next->random = th-> random -> next ;
            else th->next->random = NULL;
            th = th->next->next;
        }
        th = h;
        nh = h->next;
            while(th){ // connect the new LL next pointers and old LL next pointers too
            Node* fh =  th->next ;
            th -> next = fh -> next ;
            th = th -> next ;
            if(th) fh -> next = th -> next ;
            else  fh -> next = NULL;
        } 
        return nh;
    }
};
```

</details>

### 39. 3 sum: Find triplets that add up to a zero

<details>

> Good problem, so one way is to try all triplets and store them (answer) in a sorted fashion in a set in n^3 complexity,

> A better way is to try iterating all the pairs in `n^2` time, for the third element which will be -(a[i]+a[j]) use the hash map to check whether it exists or not, please note a very important thing i.e. since `i!=j!=k` therefore you may try using a hash map cleverly which only stores the value b/w `i` to `j` index. Please see the code to get the gist.

```
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& a) {
        set<vector<int>> an;
        int n = a.size();
        for(int i = 0; i < n; i++){
            unordered_map<int , bool > mp;
            for(int j=i+1; j<n;j++){
            int v3 = -(a[i]+a[j]);
            if(mp[v3]){
                vector<int> temp = {v3,a[i],a[j]};
                sort(temp.begin(), temp.end());
                an.insert(temp); 
            }
                mp[a[j]] =1;
            }
            mp.clear();
        }
         vector<vector<int>> ans;
        for (const std::vector<int>& v : an) ans.push_back(v);
        return ans;
    }
};
```

> it is `n^2lgm`, can we do it better like just `n^2` complexity? yes, we can do that if we sort the array and iterate the `i` pointer one by one and for every `i` we cleverly move our `j` and `k` pointers since the array is sorted, please observe that `j` and `k` pointers approach is like finding pair sum using 2 pointers approach. Please notice how we skip duplicates to boost the performance, as well as it is mandatory to do so as we need to avoid any duplicate triplets.

> similar to [Q.20](https://github.com/kuspia/Striver_SDE-shortnotes-/blob/main/README.md#20-4-sum)

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& arr) {
        int n = arr.size();
         vector<vector<int>> ans;
    sort(arr.begin(), arr.end());
    for (int i = 0; i < n; i++) {
        if (i != 0 && arr[i] == arr[i - 1]) continue; //remove duplicates:
        int j = i + 1;
        int k = n - 1;
        while (j < k) {
            int sum = arr[i] + arr[j] + arr[k];
            if (sum < 0) j++; 
            else if (sum > 0) k--;
            else {
                vector<int> temp = {arr[i], arr[j], arr[k]};
                ans.push_back(temp);
                j++;
                k--;
                //skip the duplicates:
                while (j < k && arr[j] == arr[j - 1]) j++;
                while (j < k && arr[k] == arr[k + 1]) k--;
            }
        }
    }
    return ans;
    }
};
```

</details>

### 40. Trapping Rainwater

<details>

> <img width="469" alt="Screenshot 2024-08-11 at 20 55 43" src="https://github.com/user-attachments/assets/32841d05-9169-4411-8e87-adf258897e41">

> The idea is to store the left max and right max heights and then do as shown.

```cpp
class Solution {
public:
    int trap(vector<int>& h) {
        int n = h.size();
	if (n <= 2) return 0; // Single or two heights cannot trap any water
        vector<int> l(n); //prefixMax
        vector<int> r(n); //suffixMax
        l[0] = h[0];
        for(int i =1;i<n;i++) l[i] = max ( l[i-1] , h[i]);
        r[n-1] = h[n-1];
        for(int i =n-2;i>=0;i--) r[i] = max ( r[i+1] , h[i]);
        int an=0;
        for(int i = 1 ; i < n-1 ; i++) an += max(0, min(l[i], r[i]) - h[i]);
        return an;
    }
};
```

> You are taking `2N` space can you please not take it? go through the code, now when we stand at some building what we want is min(leftMax, rightMax), so how this thing is guaranteed, this is because we always pick the smaller height buidling therefore among `leftMax` and `rightMax` by default automatically the smaller one is choosen.

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        if (n <= 2) return 0; // Single or two heights cannot trap any water
        int left = 0, right = height.size() - 1;
        int left_max = 0, right_max = 0;
        int water_trapped = 0;
        while (left <= right) {
            if (height[left] <= height[right]) {
                if (height[left] >= left_max) left_max = height[left]; // case when the building we are standing is even exceding the left_max building
                else water_trapped += left_max - height[left]; // we can store the water with help of left_max building
                ++left;
            } else {
                if (height[right] >= right_max) right_max = height[right]; // case when the building we are standing is even exceding the right_max building
                else water_trapped += right_max - height[right]; // we can store the water with help of right_max building
                --right;
            }
        }
        return water_trapped;
    }
};
```

</details>

### 41. Remove duplicates from sorted array

<details>

> n cost: easy just take two pointers pointing at index 1 now iterate the array unless `i` doesn't hit `n`, so what happens is we just check it with the previous element if `a[i] == a[i-1]` simply move on, otherwise try storing `a[i]` at `a[j]`, also in that case move both `i` and `j` pointers. Basically `j` pointer is used to store non-repeated elements, 

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& a) {
        int n = a.size();
        int k = 0;
        int i = 1, j=1;
        while(i!=n){
            if(a[i] == a[i-1]) {
                i++;
                continue;
            }else{
                a[j] = a[i];
                j++;
                i++;
            }
        }
        return j;
    }
};
```

</details>

### 42. Max consecutive ones

<details>

> The intuition is based upon the fact that whenever you encounter a 0 that means it's a breakpoint of my previously followed 1's count, also note that you may get lot of consecutive 1 series but we need to keep track of the max count, so see the code carefully to get the gist.

```cpp
class Solution {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int max_count = 0; // To store the maximum length of consecutive 1s
        int current_count = 0; // To count the current streak of 1s
        for (int num : nums) {
            if (num == 1) current_count++;
            else current_count = 0;
            max_count = max(max_count, current_count);
        }
        return max_count;
    }
};

```

</details>

### 43. N meetings in one room 

<details>

> so, the idea is to make a vector of pairs and sort it by the end time, then make the 1st meeting happen, and for the rest meeting check if the last happened meeting end time is strictly less than the meeting which is going to happen, if yes consider it else move on.

```cpp
class Solution
{
    public:
    static bool ss (  pair<int,int> &a , pair<int,int> &b ){
    return a.second < b.second ;}
    int maxMeetings(int start[], int end[], int n)
    {
vector<pair<int,int>> v;
for(int i = 0 ; i<n ; i++){
v.push_back(make_pair(start[i] , end[i]));
}
sort(v.begin() , v.end() , ss);
        int c = 1 ; // make the 1st meeting to happen 
        int prev  = v[0].second;
        for(int i =1 ;  i < n ; i++){
           if (v[i].first > prev ) {
                prev = v[i].second; // only swap if meeting is considered else skip
                c++;
            }
        }
       return c;
    }
};
```

</details>

### 44. Minimum number of platforms needed for a railway

<details>

> A implementation concept, so it's similar Q.43, but here we need to find out the minimum number of meeting rooms we have to build.
 
> We sort it by `dept` time as usual and then try iterating from index `1 to n-1`, whenever you feel a new station is needed create it and make the `dept` time of that train be stored in `ans` vector at its right place.

> `auto it = std::lower_bound(ans.begin(), ans.end(), arr);` In other words, it returns an iterator to the first element that is greater or equal to the given value, therefore we decrease it to replace the value with the new `dest` time of our current train.

> However, if u get index 0 in `it` that suggests that a new platform is needed since a Train has arrived whose arrival time is less than dept time of all other trains standing in our platforms vector `ans`.

```cpp
class Solution{
    public:
    static bool ss (  pair<int,int> &a , pair<int,int> &b ){return a.second < b.second ;}
    int findPlatform(int start[], int end[], int n)
    {
vector<pair<int,int>> v;
for(int i = 0 ; i<n ; i++){
v.push_back(make_pair(start[i] , end[i]));
}
sort(v.begin() , v.end() , ss);
vector<int> ans;
 ans.push_back(v[0].second);
 
 for(int i =1 ;  i < n ; i++){
     int arr  = v[i].first;
     int dept =  v[i].second;
     auto it = std::lower_bound(ans.begin(), ans.end(), arr);
      if (it != ans.begin()) { 
        it--; 
        *it = dept;
    }
    else{
    // case when a train has arrived and there is no platform that can me made available means
    // all platform have trains whose departure time is more than 
    // arrival time of current train
    // we need to create a new platform for that train
    sort(ans.begin(), ans.end());
    ans.insert(dept);
    }
 }
return ans.size();
}
};
```

> `O(n)` The problem of finding the minimum number of platforms required for trains at a railway station is indeed similar to the concept of locating that integer value that overlaps maximum times in all of the given ranges and printing the occurrences of that integer value, as we need to handle that time-stamp (which is our integer value) when maximum platforms are needed to handle all trains.

```cpp
class Solution{
    public:
    int findPlatform(int arrival[], int departure[], int n)
{
   int pf[2361] ;
   int requiredPlatform = 1;
   memset(pf , 0 , sizeof(pf));
   for (int i = 0; i < n; i++) {
        ++pf[arrival[i]]; 
        --pf[departure[i] + 1];
    }
     for (int i = 1; i < 2361; i++) {
        pf[i] = pf[i] + pf[i - 1]; 
        requiredPlatform = max(requiredPlatform, pf[i]);
    }
      return requiredPlatform;
}
};
```

</details>

### 45. Job sequencing problem

<details>

> Very interesting and tricky greedy problem, see the idea is that we have to pick the maximum profit job in our iteration one by one and then we need to do it at the last moment that's how we achieve maximum profit, `last` means at the deadline and if the deadline is occupied maybe try doing it before the deadline but the most delayed possible day.   

```cpp
class Solution 
{
    public:
    static bool ss (  pair<int,int> &a , pair<int,int> &b ){return a.first > b.first;}
    vector<int> JobScheduling(Job arr[], int n) 
    { 
	vector<pair<int,int>> v;
	for(int i = 0 ; i<n ; i++){
	v.push_back(make_pair( arr[i].profit  , arr[i].dead  ));
	}
	vector<int> ans1;
	sort(v.begin() , v.end() , ss);
	int pft = 0 ;
	int chk[n+1]; // stores which day is occupied with which process index id (as per sorted one not the given one)
	memset(chk,-1,sizeof(chk));
	int cnt = 0 ; // stores total number of jobs done
	for(int i = 0 ; i< v.size() ; i++){
	  if(chk[v[i].second] == -1){ //the day is free
	      pft+=v[i].first;
	      cnt++;
	      chk[v[i].second] = i; // make it occupied by the job index id 
	  }else{
	      int cc = v[i].second; // cc is the deadline day if it is 4 then that job is needed to be completed by the 4th day not necessarily on the 4th day (key point)
	      while( chk[cc] != -1){ // but on that day some job was already scheduled so we try finding a free day
	          cc--;
	          if(cc==0) break; // remember we did 1 indexing so if u reached 0 that means we can't schedule it at any cost
	      }
	      if(cc>0){ // scheduling our profitable job not exactly on the deadline but before our deadline
              pft+=v[i].first;
              cnt++;
	      chk[cc] = i;
	      }
	  }
	}
	// formatting the answer that's it
	ans1.push_back(cnt);
	ans1.push_back(pft);
	return ans1;
    } 
};
```

</details>

### 46. Fractional Knapsack

<details>

```cpp
/*
struct Item{
    int value;
    int weight;
};
*/
bool cf(Item a,Item b){
        double x = (double)a.value/a.weight;
        double y = (double)b.value/b.weight;
        return x>=y; 
}
class Solution
{
    public:
    double fractionalKnapsack(int W, Item A[], int n)
    {
        sort(A,A+n,cf);
        double p;  // answer
        for(int i=0;i<n;i++){
            if(A[i].weight<=W){ // we are taking whole units at of the items without cutting it
                W = W - A[i].weight;
                p = p + A[i].value;
            }
            else{
                p = p + W*(double)A[i].value/A[i].weight;
                break; // you can't take more now, we are totally filled 
            }
        }
        return p;
    }
        
};
```

</details>

### 47. Find minimum number of coins

<details>

> Given a value V, if we want to make a change for V Rs, and we have an infinite supply of each of the denominations in Indian currency, i.e., we have an infinite supply of { 1, 2, 5, 10, 20, 50, 100, 500, 1000} valued coins/notes, what is the minimum number of coins and/or notes needed to make the change.

```cpp
vector<int> MinimumCoins(int n)
{
    int coin[9];
    coin[0] = 1;
     coin[1] = 2;
      coin[2] = 5;
       coin[3] = 10;
        coin[4] = 20;
         coin[5] = 50;
          coin[6] = 100;
           coin[7] = 500;
            coin[8] = 1000;
            vector<pair<int,int>> v ;
            vector<int> ans ;
            for(int i = 8 ; i>=0 ; i--){
                if(n==0) break;
                v.push_back(  make_pair(n/coin[i] , coin[i]  )  ); // { how many, which coin}
                n = n%coin[i];
            }
            for(int i =0 ; i < v.size() ; i++)    while(v[i].first--) ans.push_back(v[i].second);
            return ans;
}
```


</details>

### 48. Assign cookies

<details>

> just see the code and figure out urself what is going on (easy)

```cpp
int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int child_i = 0;  
    int cookie_i = 0;
    while (child_i < g.size() && cookie_i < s.size()) {
        if (s[cookie_i] >= g[child_i]) { // If the current cookie can satisfy the current child
            child_i++;  // Move to the next child
        }
        cookie_i++;  // Move to the next cookie
    }
    return child_i;
}
```

</details>

### 49. 

<details>



```cpp
```


</details>

### 50. 

<details>



```cpp
```


</details>

### 51. 

<details>



```cpp
```


</details>



























   
