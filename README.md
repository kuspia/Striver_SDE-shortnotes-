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
            high--; // we don't move mid pointer since in next iteration we might need to swapped value from high index again back to low (case of 1) 
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
	
> Refer [Q.12](https://github.com/kuspia/Striver_SDE-shortnotes-/tree/main#12-inversion-of-array)

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

```cpp
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
   
    ans.insert(it, dept); //inserting it at begnning 
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

> Optimize it with DSU (to-do) the inner while loop 

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

### 49. Sum of all Subsets (2^n)

<details>

> Given an array print all the sum of the subset generated from it, in the increasing order.

```cpp
class Solution {
public:
    void f(const vector<int>& arr, int n, int i, vector<int>& s, int sum_val) {
        if (i == n) {
            s.push_back(sum_val);
            return;
        }
        // Take the current element
        f(arr, n, i + 1, s, sum_val + arr[i]);
        // Do not take the current element
        f(arr, n, i + 1, s, sum_val);
    }
    vector<int> subsetSums(const vector<int>& arr, int n) {
	sort(arr.begin(), arr.end()); // do it as a good practice 
        vector<int> s;
        f(arr, n, 0, s, 0);
	sort(s.begin(), s.end());
        return s;
    }
};

```

</details>

### 50. Subset II (print unique subsets)

<details>

> Given an array of integers that may contain duplicates the task is to return all possible subsets. Return only unique subsets and they can be in any order.

```cpp
class Solution {
public:
    void f(const vector<int>& nums, int i, set<vector<int>>& s, vector<int>& subset) {
        if (i == nums.size()) return; 
        // Take the current element
        subset.push_back(nums[i]);
        s.insert(subset);
        f(nums, i + 1, s, subset);
        // Backtrack
        subset.pop_back();
        f(nums, i + 1, s, subset);
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        set<vector<int>> s;
        vector<int> subset;
        // Start with the empty subset
        s.insert(vector<int>());
        // Sort the input to ensure subsets are considered in a sorted order
        sort(nums.begin(), nums.end());
        f(nums, 0, s, subset);
        // Convert the set to vector for return
        vector<vector<int>> vec(s.begin(), s.end());
        return vec;
    }
};

```

</details>

### 51. Combination Sum 1 (distinct-integers-array-sums-to-a-target-pick-elements-many-times)

<details>
	
> Given an array of distinct integers and a target, you have to return the list of all unique combinations where the chosen numbers sum to target. 


```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void helper(vector<int>& candidates, int target, int i, int curSum, vector<int> temp) {
        if (curSum == target) {
            ans.push_back(temp);
            return;
        }
        if (curSum > target || i >= candidates.size()) return;
        temp.push_back(candidates[i]);
        helper(candidates, target, i, curSum + candidates[i], temp);
        temp.pop_back();
	//ntake
        helper(candidates, target, i + 1, curSum, temp);
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<int> temp;
	sort(candidates.begin(), candidates.end()); // do it as a good practice 
        helper(candidates, target, 0, 0, temp);
        return ans;
    }
};

```

</details>

### 52. Combination Sum 2 (integers-array-sum-to-target-pick-elements-once)

<details>

```cpp
class Solution {
public:
    std::vector<std::vector<int>> ans;
    void helper(std::vector<int>& candidates, int target, int i, int curSum, std::vector<int>& temp) {
        if (curSum == target) {
            ans.push_back(temp);
            return;
        }
        if (curSum > target || i >= candidates.size()) return;
        temp.push_back(candidates[i]);
        helper(candidates, target, i + 1, curSum + candidates[i], temp);
        temp.pop_back();  
        // Skip duplicates: move to the next distinct element
        while (i + 1 < candidates.size() && candidates[i] == candidates[i + 1]) i++;
        // Recurse without including the current element
        helper(candidates, target, i + 1, curSum, temp);
    }

    std::vector<std::vector<int>> combinationSum2(std::vector<int>& candidates, int target) {
        std::vector<int> temp;
        std::sort(candidates.begin(), candidates.end()); // Sort to handle duplicates
        helper(candidates, target, 0, 0, temp);
        return ans;
    }
};

```


</details>

### 53. Palindrome Partitioning

<details>

> Return all palindromic partitions of s.

> The problem is solved by putting a bar between the characters of the string, so we start as usual and set the bar that divides the string into two halves, we check for the left half whether it is palindrome or not, if not try putting the bar to next position or recurse for the remaining right half, clearly when u reach at the position when the bar is at the end that is a successful attempt hence just print all those left substrings that u kept on chopping.

```cpp
class Solution {
public:
int isP(const string &s) {
    int left = 0;
    int right = s.size() - 1;
    while (left < right) {
        if (s[left] != s[right]) {
            return 0;
        }
        left++;
        right--;
    }
    return 1;
}
void f ( string s, int n , int bar ,  vector<vector<string>>& ans,vector<string>& str_coll     ){
        if(bar == n){
        ans.push_back(str_coll);
            return;
        }
        for(int i = bar ;  i < s.size() ; i++ ){
        string chk = s.substr( bar ,  i - bar+1) ;
        int left_half = isP(   chk   );
        if(left_half){
             str_coll.push_back(chk);
             f(s ,  n , i+1 , ans , str_coll);
             str_coll.pop_back();
        }else{
            continue; 
        }
        }
    }
    vector<vector<string>> partition(string s) {
            vector<string> str_coll;
    vector<vector<string>> ans;
        f(s , s.size() ,  0 , ans , str_coll);
        return ans;
    }
};
```

</details>

### 54. Kth permutation sequence

<details>

> Pattern based (without-recursion): suppose we have sorted numbers from 1 to n, and the factorial of every number stored, if `n` = 4 and `k` = 16, in this case what we do is we fill from left to right, now we can figure out the every number on every index, so when `n` is 4 we know we have 24 combinations, when we fix the 0th index, we are left with 6 combinations on the right hand side(1st to 3rd indices), so we have groups of size 6 for every number choosen on index 0, since k = 14, so we can say we want group 3rd (so [0] = 3), and for next iteration `k` will 4 (so we want 4th number among 6 combinations left out from (1,2,4).  

```cpp
class Solution {
public:
    string getPermutation(int n, int k) {
        vector<int> factorial(n + 1, 1);
        for (int i = 2; i <= n; ++i) factorial[i] = factorial[i - 1] * i;
        vector<int> numbers;
        for (int i = 1; i <= n; ++i) numbers.push_back(i);
        k--;
        string result;
        for (int i = 0; i < n; ++i) {
            int div = factorial[n - i - 1];
            int group = k / div;
            int rem = k % div;
            result += to_string(numbers[group]);
            numbers.erase(numbers.begin() + group);
            k = rem;
        }
        return result;
    }
};
```


> The approach involves using a counter `index` to track the number of elements currently in `current`. Within the recursion, iterate through all possible values from `1` to `n`, selecting an element if it hasn't been used yet, which is efficiently checked using a mask with `n` bits. When `index` equals `n`, it means one complete permutation has been found among all possible permutations.

```cpp
class Solution {
public:
    int count = 0;

    int findPermutation(vector<int>& v, int n, long long int& bitmask, int index, vector<int>& current, int k, vector<int>& result) {
        if (index == n) {
            count++;
            if (count == k) {
                result = current;
                return 1; // Found the k-th permutation
            }
            return 0; // Continue searching
        }

        for (int j = 0; j < n; j++) {
            long long int bit = 1 << v[j];
            if (bit & bitmask) { // If the bit is set, the number has been used, skip it
                continue;
            } else {
                current.push_back(v[j]);
                bitmask |= bit; // Mark this number as used
                if (findPermutation(v, n, bitmask, index + 1, current, k, result)) return 1; // Found the permutation
                // Backtrack
                current.pop_back();
                bitmask ^= bit; // Unmark this number
            }
        }
        return 0; // Not found
    }
    string getPermutation(int n, int k) {
        vector<int> v(n);
        vector<int> result(n);
        vector<int> current;
        for (int i = 0; i < n; i++) v[i] = i + 1;
        long long int bitmask = 0;
        findPermutation(v, n, bitmask, 0, current, k, result);
        stringstream ss;
        for (int i = 0; i < n; ++i) ss << result[i];  
        return ss.str();
    }
};

```

</details>

### 55. Print all permuation of a string or array (distinct-element-(-10to10))

<details>

> similar to K-th permutation Sequence, but here numbers can be from -10 to 10 so instead of using bitmasking we tried using the map to mark or unmark it.

```cpp
class Solution {
public:
    void f ( vector<int>& cur ,  vector<vector<int>>& ans ,vector<int>& nums, int& n, int id, 
      map<int,bool>& mp
     ){
        if(id==n){ // one of the possible permutations 
            ans.push_back(cur);
            return;
        }
        for(int i =0; i < n ; i++){
                if(  ! mp[ nums[i] ]  ) { // only pick nums[i] if it is not marked in our map mp
                        cur.push_back(nums[i]);
                        mp[ nums[i] ]  = 1 ; // mark it 
                          f(cur,ans,nums, n, id+1,mp); // call next recursion with id+1
                          cur.pop_back(); // pop the last element and then check other possibilities using the above loop
                          mp[ nums[i] ] = 0 ; // unmark that element since we have removed it from our `cur`
                }else{
                }
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<int> cur; // this stores the elements from nums and we push to ans if its size is n
        vector<vector<int>> ans; // to store final set of all permutation vectors
        int n = nums.size();
        map < int , bool > mp ; // we mark from -10 to 10 as per question to know which element has been already taken or not 
        for(int i = -10 ; i <= 10 ; i++) mp[i] = 0 ;
        f(cur,ans,nums, n, 0,mp);
        return ans;
    }
};
```

> swap algo

```cpp
void permutations(vector<vector<int>>& res, vector<int> nums, int id, int n) { 
    if (id == n) { 
        res.push_back(nums); 
        return; 
    } 
    for (int i = id; i <= n; i++) { 
        swap(nums[id], nums[i]); 
        permutations(res, nums, id + 1, n); 
        swap(nums[id], nums[i]); 
    } 
} 
vector<vector<int>> permute(vector<int>& nums) { 
    vector<vector<int>> res; 
    int n = nums.size() - 1; 
    permutations(res, nums, 0, n); 
    return res; 
} 
```


</details>

### 56. N queen problem 

<details>

> To solve this problem, place queens row by row, checking each column in the current row for a valid position. Use a helper function to validate the queen’s placement by checking the column and both diagonals. If placing a queen leads to a valid configuration, recursively attempt to place queens in the next row; if not, backtrack by removing the queen and trying the next column.

```cpp
class Solution {
public:
    // Function to check if placing a queen at position (r, c) is valid
    bool isValid(vector<vector<string>>& board, int r, int c, int n) {
        // Check the same column
        for (int i = 0; i < r; i++)
            if (board[i][c] == "Q")
                return false;
        
        // Check right upper diagonal
        for (int i = r - 1, j = c + 1; i >= 0 && j < n; i--, j++)
            if (board[i][j] == "Q")
                return false;
        
        // Check left upper diagonal
        for (int i = r - 1, j = c - 1; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == "Q")
                return false;

        return true;
    }

    // Recursive function to solve the N-Queens problem
    void solve(vector<vector<string>>& board, vector<vector<string>>& results, int n, int row) {
        if (row == n) {
            vector<string> temp;
            for (const auto& rowVec : board) {
                temp.push_back(string(rowVec.begin(), rowVec.end()));
            }
            results.push_back(temp);
            return;
        }

        for (int col = 0; col < n; col++) {
            if (isValid(board, row, col, n)) {
                board[row][col] = "Q";
                solve(board, results, n, row + 1);
                board[row][col] = "."; // Backtrack
            }
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> board(n, vector<string>(n, "."));
        vector<vector<string>> results;
        solve(board, results, n, 0);
        return results;
    }
};
```


</details>

### 57. Sudko Solver

<details>

> Interesting problem, we use two loops to iterate the cells of our board then if we encounter `.` char we try filling it with 1-9 and validate it, if we can fill it clearly we need to fill next empty cell so we call `solve` again, if we can't fill it we return 0 that shows none of them 1-9 is suitable choice for that cell hence it backtrack to previous call and on getting false it makes `board[i][j]` again to '.', then we check for rest of the possibilities.

```cpp
class Solution {
public:
    // Check if placing the number 'val' at position (r, c) is valid
    bool isValid(vector<vector<char>>& board, int r, int c, int val) {
        char ch = static_cast<char>(val + '0');
        // Check the column
        for (int i = 0; i < 9; i++) {
            if (board[i][c] == ch) return false;
        }
        // Check the row
        for (int i = 0; i < 9; i++) {
            if (board[r][i] == ch) return false;
        }
        // Check the 3x3 subgrid
        int startRow = 3 * (r / 3);
        int startCol = 3 * (c / 3);
        for (int i = startRow; i < startRow + 3; i++) {
            for (int j = startCol; j < startCol + 3; j++) {
                if (board[i][j] == ch) return false;
            }
        }
        return true;
    }

    // Backtracking function to solve the Sudoku puzzle
    bool solve(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (int k = 1; k <= 9; k++) {
                        if (isValid(board, i, j, k)) {
                            board[i][j] = static_cast<char>(k + '0');
                            if (solve(board)) return true;
                            board[i][j] = '.'; // Backtrack
                        }
                    }
                    return false; // Return false if no number fits
                }
            }
        }
        return true; // Puzzle solved
    }

    void solveSudoku(vector<vector<char>>& board) {
        solve(board); // Call the solve function
    }
};

```


</details>

### 58. M coloring problem 

<details>

> The problem is solved by attempting to color each node with one of the `m` available colors and checking if the entire graph can be colored successfully. The `isValid` function determines if a node `id` can be colored with a specific color `color` without violating constraints. During recursion, all color possibilities are explored for each node. If coloring a node fails with all `m` colors, we backtrack by resetting the color and trying the next configuration.

```cpp
class Solution {
public:
    bool isValid(vector<vector<int>>& graph, int id, int color, vector<int>& colors) {
        for (int neighbor : graph[id]) if (colors[neighbor] == color) return false;
        return true;
    }
    bool solve(vector<vector<int>>& graph, int m, int id, vector<int>& colors) {
        // If all vertices are colored
        if (id == graph.size()) return true;
        // Try coloring vertex 'id' with each color
        for (int color = 0; color < m; color++) {
            if (isValid(graph, id, color, colors)) {
                colors[id] = color;
                if (solve(graph, m, id + 1, colors)) return true;
                // Backtrack
                colors[id] = -1;
            }
        }
        return false;
    }

    bool graphColoring(bool graph[101][101], int m, int n) {
        vector<vector<int>> adjacencyList(n + 1);
        vector<int> colors(n, -1);
        // Convert adjacency matrix to adjacency list
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (graph[i][j]) {
                    adjacencyList[i].push_back(j);
                }
            }
        }
        return solve(adjacencyList, m, 0, colors);
    }
};
```

</details>


### 59. Rat in a maze

<details>

```cpp
class Solution {
public:
    void findPaths(vector<string>& ans, vector<string>& moves, vector<vector<int>>& grid, int n, int r, int c) {
        // If we've reached the bottom-right corner of the grid
        if (r == n - 1 && c == n - 1) {
            // Convert the list of moves into a single string
            string path = "";
            for (const auto& move : moves) {
                path += move;
            }
            // Add the string of moves to the answer list
            ans.push_back(path);
            return;
        }

        // Check if the current cell is blocked or visited
        if (grid[r][c] == 0) return;

        // Mark the current cell as visited
        grid[r][c] = 0;

        // Try moving left (if possible)
        if (c > 0 && grid[r][c - 1] == 1) {
            moves.push_back("L");
            findPaths(ans, moves, grid, n, r, c - 1);
            moves.pop_back(); // Backtrack
        }

        // Try moving right (if possible)
        if (c < n - 1 && grid[r][c + 1] == 1) {
            moves.push_back("R");
            findPaths(ans, moves, grid, n, r, c + 1);
            moves.pop_back(); // Backtrack
        }

        // Try moving up (if possible)
        if (r > 0 && grid[r - 1][c] == 1) {
            moves.push_back("U");
            findPaths(ans, moves, grid, n, r - 1, c);
            moves.pop_back(); // Backtrack
        }

        // Try moving down (if possible)
        if (r < n - 1 && grid[r + 1][c] == 1) {
            moves.push_back("D");
            findPaths(ans, moves, grid, n, r + 1, c);
            moves.pop_back(); // Backtrack
        }

        // Unmark the current cell as unvisited for other potential paths
        grid[r][c] = 1;
    }

    vector<string> findPath(vector<vector<int>>& grid, int n) {
        vector<string> ans; // List to store possible paths
        vector<string> moves; // List to store current moves
        findPaths(ans, moves, grid, n, 0, 0); // Start exploring paths from the top-left corner
        
        // If there are valid paths found, return them
        if (!ans.empty()) {
            return ans;
        }
        // If no valid paths are found, return a list with a single element "-1"
        return {"-1"};
    }
};

```


</details>

### 60. Word break (print all ways)

<details>

> `memo[i]` indicates whether the substring of `s` starting at index `i` can be segmented into words that are present in the `wordSet`.

```cpp

class Solution {
public:
    bool wordBreakHelper(const string& s, int start, const unordered_set<string>& wordSet, vector<int>& memo, vector<string>& result) {
        if (start == s.size()) return true;
        if (memo[start] != -1) return memo[start];

        for (int end = start; end < s.size(); ++end) {
            string substring = s.substr(start, end - start + 1);
            if (wordSet.find(substring) != wordSet.end()) {
                result.push_back(substring); // Add the substring to the result
                if (wordBreakHelper(s, end + 1, wordSet, memo, result)) {
                    return memo[start] = true;
                }
                result.pop_back(); // Backtrack and remove the substring
            }
        }
        return memo[start] = false;
    }

    pair<bool, vector<string>> wordBreak(const string& s, const vector<string>& wordDict) {
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        vector<int> memo(s.size(), -1);
        vector<string> result;

        bool canBreak = wordBreakHelper(s, 0, wordSet, memo, result);
        return {canBreak, result};
    }
};

```

</details>

### 61. The nth root of an integer

<details>

> Simple suppose you have 64 and you need to find the 3rd root of it, so traverse using BS:

> `l` = 1, `h` = 64, now check for the middle element as raised to power of 3, does it gives u `64` if yes `mid` is ur answer else reduce ur search space accordingly.

```cpp
int NthRoot(int n, int m) { // m=64, n=3
int l = 1 ;
int h =  m ;
while(l<=h){
  int mid = (l+h)/2;
  if( pow( mid , n ) == m) return mid ;
  else if(  pow ( mid , n )  > m ) h = mid-1;
  else l = mid+1 ;

}
return -1 ;
}
```

</details>

### 62. Matrix median (return l)

<details>

> The matrix is row-wise sorted, that means we can apply BS on every row, secondly: It's important to note that `mid` is a potential median, not an exact one, as it might not exist in our matrix. To tackle this, we employ a clever trick: if mid is equal to the target value, we perform a binary search on the right half as well. By doing this, we ensure that l eventually points to the exact median position if it exists.

```cpp
int bs(vector<vector<int>> &mat, int m, int n , int l , int h, int target ){
        if(l<=h){
            int mid = (l+h)/2;
            int lesser_equal = 0 ;
            for(int i = 0 ; i < m ; i++) // m rows 
            lesser_equal  += ( upper_bound( mat[i].begin() , mat[i].end()  , mid ) -   mat[i].begin()  );
            if (lesser_equal <= target)return bs(mat, m, n , mid+1 , h, (m*n)/2 );
            else return bs(mat, m, n , l , mid-1, (m*n)/2 );   
        }
        return l ;
}
int median(vector<vector<int>> &matrix, int m, int n) { // m rows and n cols 
      int ans =  bs(matrix, m, n , 1 , 1000000009, (m*n)/2 );
      return ans;
}
```

</details>

### 63. Element that appear once while others appears twice

<details>

> The idea is to compare mid with left or right wherever equality holds just try to count the remaining elements on that side (exclude pairs) if it turns out to be odd u need to recurse for that half else the other one.

> Key point: The answer will lie b/w l to h such that (h-l+1) is odd.

```cpp
class Solution {
public:
int singleNonDuplicate(vector<int>& nums) {
        int l = 0 ;
        int h = nums.size()-1;
        int m ;
        int n = nums.size();
        while( l <=h) {
            m = (l+h)/2;
            if(  m+1 <= n-1 &&  nums[m] ==  nums[m+1] ){
               if(   ( h- m +1)  & 1  ) l = m+2;
               else h = m-1; 
            }
            else if ( m-1>=0 && nums[m] == nums[m-1]){
                    if(  (    ( m - l +1 )   &1 ) ) h = m-2;    
                    else l = m+1;  
            }
            else return nums[m];
        }
return -1;
    }
};
```

</details>

### 64. Search Element in a Rotated Sorted Array


<details>

> Rember that when u find a mid, then if (nums[m] <= nums[h])  -> Right half is sorted, and the pivot is in the left half, and vice-versa

> Now that u know if the right half is sorted then try checking target lies in the right range or not, if yes, that tells that we need to recur for the right half again else the left half. 

```cpp
class Solution {
public:
    int f(vector<int>& nums, int target, int l, int h) {
        if (l <= h) {
            int m = (l + h) / 2;
            if (nums[m] == target) return m;
            if (nums[m] <= nums[h]) {
                // Right half is sorted, and the pivot is in the left half.
                if (target >= nums[m] && target <= nums[h]) {
                    return f(nums, target, m+1 , h );
                } else {
                    return f(nums, target, l , m-1 );
                }
            }
            if (nums[m] >= nums[l]) {
                // Left half is sorted, and the pivot is in the right half.
                if (target >= nums[l] && target <= nums[m]) {
                    return f(nums, target, l , m-1 );
                } else {
                    return f(nums, target, m+1 , h );
                }
            }
        }
        return -1;
    }
    int search(vector<int>& nums, int target) {
        return f(nums, target, 0, nums.size() - 1);
    }
};
```

</details>

### 65. Median of 2 sorted array

<details>

> The question is very tricky so what we do is I make `nums2` array as my smaller always (not mandatory just to reduce time complexity), now idea is that we can pick `0` to `n2` elements from `nums2` over which we do the BS and rest we pick from `nums1` array denoted by `m1` in our code. Now what happens is we choose m1 as `(n1+n2)/2 - m2` because we want to take half of the elements from both arrays when combined together to locate my median, however, we use even/odd case rules to handle it, especially in my returning situation that u may see in the code. Please note if my `m2` is 3 that means we choose `3` elements marked by `nums2` from index `0` to `2` (inclusive). 

> Remember l1 l2 r1 r2 are chosen and defined with default value in case the index is not found, these default values actually helps you to ignore a particular array if we hit at some index that is out of bound and are useful for cases like: [1,3] [2] 

```cpp
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
     int n1 = nums1.size();
     int n2 = nums2.size();
     if(n2>n1) return findMedianSortedArrays(nums2 , nums1);
     // do BS on nums2 size 
     int l ,h, m2, m1 ;
     //m2 tells how many elements we pick from nums2 and same for m1
     l = 0;
     h = n2 ; // i can pick atmost n2 elemetns from nums2
     int cnt =0 ;
     while(l<=h){
            m2 = (l+h)/2;
            m1 = (n1+n2)/2 - m2; 
            int l1 ,l2 , r1 , r2 ;
            l1 = (m1 - 1 >= 0 && m1 - 1 < n1) ? nums1[m1 - 1] : -1000000;
            l2 = (m2 - 1 >= 0 && m2 - 1 < n2) ? nums2[m2 - 1] : -1000000;
            r1 = (m1 >= 0 && m1 < n1) ? nums1[m1] : 1000000;
            r2 = (m2 >= 0 && m2 < n2) ? nums2[m2] : 1000000;
            if( r2>=l1 && r1 >= l2 ) 
            {
                    if  (  (n1+n2) & 1 )  return min (r1 , r2 ) * 1.0;
                    else return (max ( l1 , l2) + min ( r1 , r2 ) )/2.0;    
            }
            if (l1 > r2) 
            l = m2 + 1;
            else if (l2 > r1) 
            h = m2 - 1;
     }
return -1.0; //dummy
    } 
};
```

</details>

### 66. Kth element of 2 sorted array

<details>

> similar to finding the median of two sorted arrays but here remember as you want to find the kth element we need to fill our left half with (k-1) element and also if the optimal condition is met we can `return min (r1, r2 )`, remember the trick part is: `l` =  `max (k-1 - n1 ,0)` `h` = `min(n2,k-1)`

```cpp
class Solution{
    public:
    int kthElement(int nums1[], int nums2[], int n1, int n2, int k)
    {
     if(n1 < n2 ) return kthElement (  nums2 , nums1 , n2 , n1 , k) ;
     int l ,h, m2, m1 ;
     l =  max ( (k-1) - n1 ,0) ;
     h = min (  n2, (k-1));
     while(l<=h){
            m2 = (l+h)/2;
            m1 = (k-1) - m2; 
            int l1 ,l2 , r1 , r2 ;
            l1 = (m1 - 1 >= 0 && m1 - 1 < n1) ? nums1[m1 - 1] : INT_MIN;
            l2 = (m2 - 1 >= 0 && m2 - 1 < n2) ? nums2[m2 - 1] : INT_MIN;
            r1 = (m1 >= 0 && m1 < n1) ? nums1[m1] : INT_MAX;
            r2 = (m2 >= 0 && m2 < n2) ? nums2[m2] : INT_MAX;
            if( r2>=l1 && r1 >= l2 ) return min (r1 , r2 ) ;
            if (l1 > r2) 
            l = m2 + 1;
            else if (l2 > r1) 
            h = m2 - 1;
     }
     return 1 ; //dummy
    }
};
```

</details>

### 67. Allocate minimum number of pages (return l)

<details>

> Initialize `low` and `high` as shown in code, not always `1` to `1e9`  (always for any BS problem ur search space is to be chosen wisely else u will get WA. Reason, imagine, for this, to get the gist `[12, 13, 14]` `n=3` or `n=1`

> When you find an optimal answer as we need to shrink it so again we look into the left half for that, i.e. suppose my max of min was 72 which is optimal so we still want to reduce so in case of equality try moving towards left side doing so, u end up `low` as your answer.

> Try initializing `required_students` as 1 even before u start else it won't work properly because when u go in the else block where u increment `cnt` it actually counts the next student, not the previous one.

```cpp
class Solution 
{
public:
    // Function to find the minimum number of pages.
    int findPages(int a[], int n, int m) 
    {
        if (m > n) return -1;  // More students than books is not possible.
        int low = *max_element(a, a + n);
        int high = accumulate(a, a + n, 0);
        while (low <= high) {
            int mid = (low + high) / 2;
            int required_students = 1;
            int current_sum = 0;
            for (int i = 0; i < n; i++) {
                if (current_sum + a[i] <= mid) {
                    current_sum += a[i];
                } else { // this is for next student (tbh)
                    required_students++;
                    current_sum = a[i];
                }
            }
            if (required_students > m) low = mid + 1;
	    else high = mid - 1;
        }
        return low;
    }
};
```
</details>

### 68. Aggresive cows
<details>
	
> This question secretly uses BS and often seems as the problem of recursion where we want to explore all possibilities, so the idea is u assume the distance from `l` to `h` and reduce it by BS, what we check in `n` time, we check can we seriously place at least `k` cows with minimum distance `m` b/w any two cows, if yes we need to update our answer and we go into right half to find more possible distance between two cows however if we fail to do so that means we need to look in left half as we have chosen much more than expected maximum value of the minimum distance between any two cows.  

```cpp
int aggressiveCows(vector<int> &stalls, int k) {
    int l = 1;
    sort(stalls.begin(), stalls.end());
    int h = stalls.back() - stalls.front(); // Max possible distance
    int result = 0;
    while (l <= h) {
        int m = l + (h - l) / 2;
        int cnt = 1;  // Place the first cow in the leftmost stall
        int prev = stalls[0];
        for (int i = 1; i < stalls.size(); i++) {
            if (stalls[i] - prev >= m) {
                cnt++;
                prev = stalls[i];
            }
        }
        if (cnt >= k) {
            result = m;  // Update result and continue searching for larger distances
            l = m + 1;
        } else {
            h = m - 1;
        }
    }
    return result;
}
```

</details>

### 63. Max heap and Min heap Implementation

<details>

> The basic idea is heap stores the elements from left to right because of which for `0` indexing for any node `i`, the parent is `(i-1)/2`, and the left and right child is calc as done in the code, so let's talk about insertion which says insert at last then keep on climbing towards the root and swap with the parent if necessary, when you have to delete min key make sure last element should take the place of 0th index and then heapify it by comparing the left/right children recursively from the root node, we initially assume that my smallest child is parent index but swap with smallest child's indices, in case you did the swapping again call heapify function.

```cpp
void insert (vector<int>& mi, int val) {
mi.push_back(val);
int id = mi.size() -1 ;
int par = (id-1)/2;
while( mi[par] > mi[id] ){
    swap (mi[par] , mi [id]);
    id = par;
    par = (id-1)/2;
}
}
void heapify(vector<int>& mi, int p) {
    int smallest = p; // Initialize the smallest element as the parent
    int l = 2 * p + 1; // Left child index
    int r = 2 * p + 2; // Right child index
    // Check if the left child is smaller than the parent
    if (l < mi.size() && mi[l] < mi[smallest])   smallest = l;
    if (r < mi.size() && mi[r] < mi[smallest])  smallest = r;
    // If the smallest element is not the parent, swap and recursively heapify
    if (smallest != p) {
        swap(mi[p], mi[smallest]);
        heapify(mi, smallest);
    }
}
vector<int> minHeap(int n, vector<vector<int>>& q) {
    vector<int> mi ; // min heap DS
    vector<int> ans;
    for(int i=0;i<n;i++){
        if(q[i][0]==0) // case of insertion 
        {
		insert(mi , q[i][1]);
        }else{
            ans.push_back(mi[0]); //buiding our answer array
            // now pop mi[0] and heapify
            mi[0] = mi[mi.size()-1];
            mi.pop_back();
            heapify(mi,0);
        }
    }
    return ans;
}
```

</details>

### 65. Kth largest element


<details>
	
> You can do it with the median of medians, I will post the solution however you can also try doing it with the heap concept of size k, where you need to maintain the size of the heap = k, and when you finish iterating the elements the top is your answer.

> **K-th Largest → Min-Heap**: **K-th Smallest → Max-Heap**: Take opposite DS 
 
```cpp
class Solution {
public:
std::vector<int> help(std::vector<int>& nums) {
    std::vector<int> result;
    // Divide the nums into groups of 5
    for (size_t i = 0; i < nums.size(); i += 5) {
        size_t groupSize = std::min(static_cast<size_t>(5), nums.size() - i);
        // Sort the group
        std::sort(nums.begin() + i, nums.begin() + i + groupSize);
        // Find the middle element index
        size_t middleIndex = i + (groupSize - 1) / 2;
        // Add the middle element to the result
        result.push_back(nums[middleIndex]);
    }
    return result;
}
    int mom ( vector<int>& nums, int k ){
    vector< int > v = help(nums);
    while(v.size() != 1){
        v = help(v);
    }
    int pvt = v[0];
    int i = 0 ;
    int c = 0 ; // in case u encounter a pivot just count it
    int j = nums.size() -1 ;
    // partition algo on basis of my pivot element
    vector< int > temp_nums ( nums.size() ) ;

    for(int k = 0 ; k < nums.size() ; k++){
        if(nums[k] > pvt){
            temp_nums[i] = nums[k];
            i++;
        }
       else if(nums[k] < pvt){
            temp_nums[j] = nums[k];
            j--;
        }
      else  c++;
    }
    while(c--){
        temp_nums[i] = pvt;
        ++i;
    }
    i--; // position of pvt (means it is sorted position of that number, please remember)
        if (i == k - 1) return pvt;
        else if (i > k - 1) {
            vector<int> left_half(temp_nums.begin(), temp_nums.begin() + i);
            return mom(left_half, k);
        } else {
            vector<int> right_half(temp_nums.begin() + i + 1, temp_nums.end());
            return mom(right_half, k - i - 1);
        }
    return -1;
    }
    int findKthLargest(vector<int>& nums, int k) {
        return mom ( nums , k ) ;
    }
};
```

</details>

### 66. Maximum sum combination (two-array-n-size-c-max-sum-combinations)

<details>

> Very interesting problem and a tough one, `c` is at most `n`, now listen we sort and iterate both arrays from back we push the last elements sum and their indexes in the set (we need to maintain the unique indexes pairs as sometimes there might be duplications while you are applying the algorithm), now key point to observe is that if my `(x,y)` is the biggest pair the next biggest pair will be `(x-1,y)` or `(x,y-1)` you might observe it very easily (use maths, assume the varibales as say, `a1`, `a2`, `a3` and `b1`, `b2`, `b3`, then apply conditions) but when you enter the loop next time we need to pick that pair which was greater among (x,y-1) or (x-1,y), so we maintain max-heap `pq` because the top element has maximum sum pair every time, please notice the next two possible max sum pairs can be generated from the maximum sum pair which can be found at top of the `pq` at any point of the time.

```cpp
vector<int> Solution::solve(vector<int> &a, vector<int> &b, int c) {
        int n = a.size();
        sort(a.begin(), a.end());
        sort(b.begin(), b.end());
        priority_queue<pair<int, pair<int, int>>> pq; //max-heap (sum, (i, j))
        set<pair<int, int>> s;
	//take the last pair from both arrays 
        pq.push(make_pair(a[n - 1] + b[n - 1], make_pair(n - 1, n - 1)));
        s.insert({n - 1, n - 1});
        vector<int> ans;
        while (c > 0) {
            pair<int, int> cur = pq.top().second;
            int x = cur.first;
            int y = cur.second;
            ans.push_back(pq.top().first);
            pq.pop();
            c--;
            if (y - 1 >= 0 && s.find({x, y - 1}) == s.end()) {
                s.insert({x, y - 1});
                pq.push(make_pair(a[x] + b[y - 1], make_pair(x, y - 1)));
            }
            if (x - 1 >= 0 && s.find({x - 1, y}) == s.end()) {
                s.insert({x - 1, y});
                pq.push(make_pair(a[x - 1] + b[y], make_pair(x - 1, y)));
            }
        }
        return ans;
}
```

</details>

### 67. Median from Data Stream

<details>

> <img width="526" alt="Screenshot 2024-08-15 at 14 34 22" src="https://github.com/user-attachments/assets/81039e8b-64f0-44cb-8af5-b3da0d5231c9">

> Very interesting question again that uses two heaps max and min at the same time, so imagine that you have an increasing sequence as shown: `________________________>` now u know the median is located at the center so let's break it `___________>____________>` let's remodify it as shown:

> <img width="104" alt="Screenshot 2024-08-15 at 14 37 17" src="https://github.com/user-attachments/assets/97ba35dc-2eaf-45ca-b8d2-973a71fcec5b">

```
1 2 3 4 5 6 : 
maxheap minheap
3 	4
2 	5
1 	6
```

> So when we process the elements one by one we can try this structure using max heap on the left while min heap on the right, so at this point you have two numbers which we place manually with `if-else` logic and after that when you have further numbers from the stream you can just place it in a desired heap and make sure you `balance` it because we are looking for `median` that's why both structures should have equal or at most a difference of one element.


```cpp
class MedianFinder {
public:
    priority_queue<int> ma; // Max-heap for the lower half
    priority_queue<int, vector<int>, greater<int>> mi; // Min-heap for the upper half
    void addNum(int num) {
        if (ma.empty() || num <= ma.top()) ma.push(num);
        else mi.push(num);
        // Balance the heaps if necessary, at most size diff. of unity 
        if (ma.size() > mi.size() + 1) {
            mi.push(ma.top());
            ma.pop();
        } else if (mi.size() > ma.size()) {
            ma.push(mi.top());
            mi.pop();
        }
    }
    double findMedian() {
        if (ma.size() > mi.size()) return ma.top();
        else return (ma.top() + mi.top()) / 2.0;
    }
};
```

</details>

### 68. Merge K-sorted arrays

<details>


```cpp
```

</details>


### 69. K most frequent elements

<details>


```cpp
```

</details>

### 70.

<details>


```cpp
```

</details>














