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

> Refer Q.20

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

### 69. Max heap and Min heap Implementation

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

### 70. Kth largest element


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

### 71. Maximum sum combination (two-array-n-size-c-max-sum-combinations)

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

### 72. Median from Data Stream

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

### 73. Merge K-sorted arrays

<details>

> Again an outstanding problem see here you learn to create a min-heap with pair of `(int, pair of (int, int))` -> `(value_of_array, (array_number, index_of_array))`, see what intuition is, try maintaining the min heap size as k always, and for 0th index preprocess it before the loop, now we say unless pq is not empty we get the top element and pop it and then again we take the next element from that array from which we found the top most minimum value, we can do so because we have advance `pq` min-heap-DS-templating.

> If my top element is this then `{ 4, { 3, 10} }` this tells us that the minimum value as far from all k arrays is 4 and it is from the 3rd array whose current pointer is at location 10 [0 indexing is being followed remember].

> If the condition `if (ka[cur.second.first].size() != cur.second.second + 1)` fails, it means that the current array has been fully iterated over, and no more elements are left to push into the priority queue. When this happens for all `k` arrays, the loop exits, as there are no more elements to process.

```cpp
struct Compare {
    bool operator()(const pair<int, pair<int, int>>& a, const pair<int, pair<int, int>>& b) {
        return a.first > b.first; // Reverse comparison for min-heap
    }
};
vector<int> mergeKSortedArrays(vector<vector<int>>& ka, int k) {
    priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, Compare> pq; //min-heap DS 
    // priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> pq; // this is fine too
    // Initialize the heap with the first element of each array
    for(int i = 0; i < k; i++) if (!ka[i].empty()) pq.push({ka[i][0], {i, 0}}); 
    vector<int> v;
    while(!pq.empty()) {
        auto cur = pq.top();
        pq.pop();
        v.push_back(cur.first);
        // If there's another element in the same array, add it to the heap
        if(ka[cur.second.first].size() != cur.second.second + 1) {
            pq.push({ka[cur.second.first][cur.second.second + 1], {cur.second.first, cur.second.second + 1}});
        }
    }
    return v;
}
```

</details>


### 74. K most frequent elements

<details>

> Since the elements are distinct, we maintain a map `mp` where each element is paired with its frequency (`element, count`). We then push these pairs into a max heap (priority queue). Once the map is populated, we pop from the max heap `K` times to retrieve the `K` most frequent elements.


```cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        map<int,int> mp;
        priority_queue <  pair<int,int> > pq; //max heap
        for(int i = -1e4 ; i<= 1e4 ; i++ ) mp[i]=0;
        for(auto num : nums) mp[num]++;
        for(int i = -1e4 ; i<= 1e4 ; i++ ) if(mp[i]) pq.push({mp[i],i});
         vector<int> v;
         while(k--){
             v.push_back(pq.top().second);
             pq.pop();
         }
         return v;
    }
};
```

</details>

### 75. Stack using arrays

<details>

```cpp
class Stack {
public:
    int  a[1000000], t , ma;
    Stack(int c) { // c is maximum size 
        t=-1;
        ma = c; // ma-1 will be the max limit up to which `t` can go 
    }
    void push(int num) {
        if(t<ma -1)
        a[++t] = num;  
    }
    int pop() {
        if(t>=0 && t<ma) return a[t--];
        return -1;
    }
    int top() {
        if(t>=0 && t<ma) return a[t];
        return -1;
    }
    int isEmpty() {
        return t == -1 ? 1:0;
    }
    int isFull() {
        return t == ma-1 ? 1:0;
    }
};
```

</details>

### 76. Queue using arrays 

<details>
	
```cpp
class Queue {
	int f, r;
	vector<int> arr;
        public:
	Queue()
	{
		f = 0;
		r = 0;
		arr.resize(100001);
	}
	void enqueue(int e)
	{
		arr[r++] =e;
	}
	int dequeue()
	{
		if(f<r)	return arr[f++];
		else return -1; 
	}
};
```

</details>

### 77. Implement Stack using a single queue

<details>

> Remember to take the `ip`, and `op` queue and always remember that my push operation will take n cost while the other two are of O(1) cost, why is this algo working which we have written? 

> <img width="458" alt="Screenshot 2024-08-15 at 17 03 59" src="https://github.com/user-attachments/assets/3a0486d7-3f89-42fe-9129-01dbba9dd6ee">

> if in a hurry, skip this and read another way when we just use a single queue

```cpp
class MyStack {
public:
    queue <int> ip,op;
    void push(int x) { // n cost operation always 
        ip.push(x);
        while(op.size()!=0){
            int temp = op.front();
            ip.push(temp);
            op.pop();
        }
        while(ip.size()!=0){
            int temp = ip.front();
            op.push(temp);
            ip.pop();
        }

    }
    int pop() {
        int ans = op.front();
        op.pop();
        return ans;
    }
    int top() {
        int ans = op.front();
        return ans;
    }
    bool empty() { return ip.size() + op.size() ? 0 : 1 ; }
};
```

> Using a single queue, here we also need to have `n` cost for push operation again, so what we do is push the element in the queue and for op.size()-1 times push then pop in the `op` queue using recursion.

> <img width="640" alt="Screenshot 2024-08-15 at 17 20 17" src="https://github.com/user-attachments/assets/795ee3cd-d831-4330-bbc7-059c71135a1d">


```cpp
class MyStack {
public:
    queue <int> op;
    void f(queue<int>& op, int swap){
        if(swap==0) return;
        int temp = op.front();
        op.push(temp);
        op.pop();
        f(op, swap-1);
    }
    void push(int x) { // n cost operation always 
        op.push(x);
        int swap = op.size()-1;
        f(op,swap);
    }
    int pop() {
        int ans = op.front();
        op.pop();
        return ans;
    }
    int top() {
        int ans = op.front();
        return ans;
    }
    bool empty() { return op.size() ? 0 : 1 ; }
};
```

</details>

### 78. Implement Queue using stack 

<details>

> Use two stack `ip`/`op`, make push operation `O(1)` and push always in `ip` stack while for `top()` and `pop()` operations always remember if op.size() == 0 transfer content of ip -> op and then you can enjoy the pop() and top() operation cost as O(1) unless op.size() == 0 again, the only pain that happens is O(n) cost when u transfer ip->op

> <img width="675" alt="Screenshot 2024-08-15 at 17 27 51" src="https://github.com/user-attachments/assets/91c42d66-3e0a-44e1-9e26-646301538514">

```cpp
class MyQueue {
public:
    stack<int> ip,op;
    void push(int x) { //let's make push as O(1), best way we did 1 cost operation as of now yoyo 
        ip.push(x);
    }
        int pop() { 
        if(op.size()==0) {
            while(ip.size()!=0){
                op.push(ip.top());
                ip.pop();
            }
        }
        int ans = op.top();
        op.pop();
        return ans;
    }
    int peek() {
        if(op.size()==0) {
            while(ip.size()!=0){
                op.push(ip.top());
                ip.pop();
            }
        }
        int ans = op.top();
        return ans;
    }
    bool empty() {  return ip.size() + op.size() ? 0 : 1 ;   }
};
```

</details>

### 79. Check for balanced parenthesis

<details>


```cpp
class Solution {
public:
    bool isValid(std::string s) {
        std::stack<char> stk;
        for (char c : s) {
            if (c == '(' || c == '{' || c == '[') {
                stk.push(c);
            } else {
                if(stk.size() == 0 ) return 0;
                int f = 0 ;
                if (c == ')' && stk.top() == '(') {
                    f=1;
                    stk.pop();
                }
                if (c == '}' && stk.top() == '{') {
                     stk.pop();
                f=1;
                }
                if (c == ']' && stk.top() == '[') {
                    f=1;
                     stk.pop();
                }
                if(!f) return 0;
            }
        }

        return stk.size() == 0 ? 1 : 0;
    }
};
```

> Bonus question for u, hehe it was very easy right u thought to move on now and see this too, HAHAHA, no easy escapes

```cpp
void generateParenthesisHelper(int n, int open, int close, string current, vector<string> &result) {
    if (current.length() == 2 * n) {
        result.push_back(current);
        return;
    }
    if (open < n) generateParenthesisHelper(n, open + 1, close, current + "(", result);
    if (close < open)  generateParenthesisHelper(n, open, close + 1, current + ")", result);
    
}
vector<string> generateParenthesis(int n) {
    vector<string> result;
    generateParenthesisHelper(n, 0, 0, "", result);
    return result;
}
```

</details>

### 80. NGE Next greater element 

<details>

> NGE code: see we try to push the indexes of the elements in my stack, but if the upcoming element is greater than that top of stock that means it is my NGE for the top position index so we store it and we keep on checking it for next top elements of my stack unless `s.size() > 0 &&   arr[s.top()] < arr[i]` else push the element index in your stack.

```cpp
class Solution
{
    public:
    vector<long long> nextLargerElement(vector<long long> arr, int n){
        vector<long long> v( arr.size() , -1 );
        stack<int >  s ; // stores the indexes
        for(int i = 0; i < arr.size() ; i++){
            while(      s.size() > 0 &&   arr[s.top()] < arr[i] ){
               v[s.top()] = arr[i];
               s.pop();
            }
            s.push(i);
        }
        return v;
    }
};
```
> [Problem](https://leetcode.com/problems/next-greater-element-i/description/) solution:

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        int n1 = nums1.size() ;
        int n2 = nums2.size();
        map < int , int > mp ;
        for(int i=0 ;  i<= 1e4 ; i++) mp[i] = -1 ;
        
        stack <int > s ;
        for(int i=0 ;  i< n2 ; i++){
            while(s.size() > 0 && nums2[i] > nums2[s.top()]){
                mp[nums2[s.top()]] = i;
                s.pop();
            }
            s.push(i);
        }
        vector<int> an;
        for(int i=0 ;  i< n1 ; i++) {
         if (  mp[  nums1[i]  ] != -1 ) an.push_back(      nums2[mp[nums1[i]]]      );
         else an.push_back(-1);
     }
         return an;
    }
};
```

</details>

### 81. Sort a stack 

<details>


> Interesting question! The approach involves using two recursive functions, `f` and `f2`. The function `f` is responsible for popping all the elements from the stack, while `f2` handles placing each popped element back in the correct position.

> Here's how it works: Imagine you pop all the elements from the stack using `f`, leaving the stack empty. Then, you start inserting the elements back one by one using `f2`. When inserting an element in `f2`, you check if the stack is empty or if the top element of the stack is less than or equal to the current element (`if (s.size() == 0 || s.top() <= val)`). If this condition is true, you simply push the element back onto the stack. If the condition is false, you pop the top element and make another recursive call to `f2` to compare the current element with the elements below it. The moment the base condition of `f2` is met, the current element will be inserted into its correct position. As you backtrack through `f2`, the elements that were previously popped out are pushed back onto the stack. Throughout this process, the key is to maintain the sorted order of the stack while working inside `f2`.

---

```cpp
void f2(stack<int>& s , int val){
	if (s.size()== 0 || s.top() <= val ) {
		s.push(val);
		return;
	}
	int val1 = s.top();
	s.pop();
	f2(s , val);
	s.push(val1);
}
void f(stack<int>& s){
if(s.size()==0) return ;
int val = s.top();
s.pop();
f(s);
f2(s,val);
}
```

</details>

### 82. NSE Next smaller element 

<details>

```cpp
vector<int> Solution::prevSmaller(vector<int> &a) {
    int n = a.size();
    reverse(a.begin(), a.end());
    vector<int> an(a.size(), -1);
    stack<int> s;
    for (int i = 0; i < n; i++) {
        while (!s.empty() && a[i] < a[s.top()]) {
            an[s.top()] = a[i];
            s.pop();
        }
        s.push(i);
    }
    reverse(an.begin(), an.end());
    return an;
    
}
```

</details>

### 83. LRU cache (least recently used)

<details>

> This is a very interesting problem that involves using a doubly linked list (DLL) and testing your understanding of pointers and memory management. The key idea is to create a head and tail node in the DLL and do further process.

**Operation: `get`**
- **Check**: Is the key available?
  - **Yes**: Return the value and move the node to the front.
  - **No**: Return `null`.

**Operation: `put`**
- **Check**: Is the key already present?
  - **Yes**: Update the value and move the node to the front.
  - **No**:
    - **Check**: Does the current size allow for a new node?
      - **Yes**: Insert the new node at the beginning.
      - **No**: Delete the least recently used (LRU) node, then insert the new node at the beginning.

> <img width="550" alt="Screenshot 2024-08-15 at 19 12 09" src="https://github.com/user-attachments/assets/65be7e49-c1e9-4a4e-80d1-cf4f876867d6">

```cpp
class LRUCache {
public:
    int cur_size = -2;
    struct node {
        int val,key;
        struct node *prev , *next ;
    };
    struct node* create(int key, int val){
        cur_size++;
        struct node* temp = new node ();
        temp->key = key;
        temp-> val = val ;
        temp->prev = nullptr;
        temp->next = nullptr;
        return temp; 
    }
    struct node* insert_front (int key , int val){
            struct node* temp = create(key, val);
            struct node* head_next = head->next;
            head->next = temp;
            temp->next = head_next;
            head_next->prev = temp;
            temp->prev = head;
            return temp;
    }

    void del_node(struct node* delit){
        cur_size--;
        struct node* temp2 = delit->next;
        struct node* temp1 = delit->prev;
        temp1->next = temp2;
        temp2->prev = temp1;
        delete delit;
    }
    int ma;
    map <int , struct node*> mp; //key and node address
    struct node *head, *tail;
    LRUCache(int capacity) {
        ma = capacity;
        head = create (-1,-1);
        tail = create (-1,-1);
        head -> next = tail;
        tail -> prev = head;
    }
    int get(int key) {
        int val = -1;
        if (mp.find(key) != mp.end()) {
            struct node* add = mp[key];
            val = add->val;
            del_node(add); 
            struct node* firstnode = insert_front(key ,val);
            mp[key] = firstnode;
        }
        return val;
    }
    void put(int key, int value) {
        if (mp.find(key) != mp.end()) {
            struct node* add = mp[key];
            del_node(add);
            struct node *firstnode = insert_front(key , value);
            mp[key] = firstnode;
        }else{
            if (cur_size == ma) {
                mp.erase(tail->prev->key);
                del_node(tail->prev);
            }    
            struct node *firstnode = insert_front(key ,value);
            mp[key] = firstnode;
        } 
    }
};
```

</details>

### 84. LFU cache

<details>

### Operation: `get`

1. **Check:** Is the key available in the `keyNode` map?
   - **Yes:**
     - Retrieve the node associated with the key.
     - Return the node's value.
     - Update the frequency of the node:
       - Remove the node from its current frequency list.
       - Increase the node's frequency by 1.
       - Add the node to the front of the list corresponding to its new frequency.
       - Update the `minFreq` if necessary (i.e., if the list for the current minimum frequency is now empty).
   - **No:**
     - Return `-1`, indicating the key is not present in the cache.

### Operation: `put`

1. **Check:** Is the key already present in the cache?
   - **Yes:**
     - Retrieve the node associated with the key.
     - Update the node’s value with the new value.
     - Update the frequency of the node:
       - Remove the node from its current frequency list.
       - Increase the node's frequency by 1.
       - Add the node to the front of the list corresponding to its new frequency.
       - Update the `minFreq` if necessary.

   - **No:**
     - **Check:** Is the cache at its maximum capacity (`curSize == maxSizeCache`)?
       - **Yes:**
         - Find the list corresponding to the `minFreq`.
         - Remove the least recently used (LRU) node from this list (this is the node at the tail's previous position).
         - Delete this node's entry from `keyNode`.
         - Decrease the current cache size (`curSize--`).
       - **No:** No need to remove any nodes; proceed to the next step.

     - **Insert the new node:**
       - Create a new node with the given `key` and `value` and set its frequency to 1.
       - Insert this new node at the front of the list corresponding to frequency 1.
       - Add this new node to the `keyNode` map.
       - Update the `freqListMap` to include this node in the list for frequency 1.
       - Set `minFreq` to 1 since a new node with the lowest frequency is added.
       - Increase the current cache size (`curSize++`).

```cpp
struct Node {
    int key, value, cnt;
    Node *next; 
    Node *prev;
    Node(int _key, int _value) {
        key = _key;
        value = _value; 
        cnt = 1; 
    }
}; 
struct List {
    int size; 
    Node *head; 
    Node *tail; 
    List() {
        head = new Node(0, 0); 
        tail = new Node(0,0); 
        head->next = tail;
        tail->prev = head; 
        size = 0;
    }
    
    void addFront(Node *node) {
        Node* temp = head->next;
        node->next = temp;
        node->prev = head;
        head->next = node;
        temp->prev = node;
        size++; 
    }
    
    void removeNode(Node* delnode) {
        Node* delprev = delnode->prev;
        Node* delnext = delnode->next;
        delprev->next = delnext;
        delnext->prev = delprev;
        size--; 
    }
    
};
class LFUCache {
    map<int, Node*> keyNode; 
    map<int, List*> freqListMap; 
    int maxSizeCache;
    int minFreq; 
    int curSize; 
public:
    LFUCache(int capacity) {
        maxSizeCache = capacity; 
        minFreq = 0;
        curSize = 0; 
    }
    void updateFreqListMap(Node *node) {
        keyNode.erase(node->key); 
        freqListMap[node->cnt]->removeNode(node); 
        if(node->cnt == minFreq && freqListMap[node->cnt]->size == 0) {
            minFreq++; 
        }
        
        List* nextHigherFreqList = new List();
        if(freqListMap.find(node->cnt + 1) != freqListMap.end()) {
            nextHigherFreqList = freqListMap[node->cnt + 1];
        } 
        node->cnt += 1; 
        nextHigherFreqList->addFront(node); 
        freqListMap[node->cnt] = nextHigherFreqList; 
        keyNode[node->key] = node;
    }
    
    int get(int key) {
        if(keyNode.find(key) != keyNode.end()) {
            Node* node = keyNode[key]; 
            int val = node->value; 
            updateFreqListMap(node); 
            return val; 
        }
        return -1; 
    }
    
    void put(int key, int value) {
        if (maxSizeCache == 0) {
            return;
        }
        if(keyNode.find(key) != keyNode.end()) {
            Node* node = keyNode[key]; 
            node->value = value; 
            updateFreqListMap(node); 
        }
        else {
            if(curSize == maxSizeCache) {
                List* list = freqListMap[minFreq]; 
                keyNode.erase(list->tail->prev->key); 
                freqListMap[minFreq]->removeNode(list->tail->prev);
                curSize--; 
            }
            curSize++; 
            // new value has to be added who is not there previously 
            minFreq = 1; 
            List* listFreq = new List(); 
            if(freqListMap.find(minFreq) != freqListMap.end()) {
                listFreq = freqListMap[minFreq]; 
            }
            Node* node = new Node(key, value); 
            listFreq->addFront(node);
            keyNode[key] = node; 
            freqListMap[minFreq] = listFreq; 
        }
    }
};
```

</details>

### 85. Largest rectangle in the histogram 

<details>

> We start by standing at the `i`th bar of the histogram and look both to the left and right to find the indices of the nearest smaller elements (NSE) on either side. We store these indices in arrays `l[]` (for the left side) and `r[]` (for the right side). Since the next smaller element can't contribute to the maximum area, we adjust the right array by decrementing the index of the NSE. Similarly, for the left array, we iterate from `n-1` to `0` and increment the NSE index, as it won't be part of the area either. 

> When considering the `i`th position, we always include it in our potential maximum area calculation and try to extend the area as much as possible in both directions. If we encounter `-1` in the left or right arrays, we take the boundaries of the bar graph, which are the `0` and `n-1` indices, respectively.

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& h) {
        int n = h.size();
        vector<int> r(n,-1);
         vector<int> l(n,-1);
          stack<int> s;
         for(int i=0;i<n;i++){
             while(s.size() > 0 && h[s.top()] > h[i] ){
            r[s.top()] = i-1;
            s.pop();
             }
            s.push(i);
         }
         while(s.size()!=0) s.pop();
         for(int i=n-1;i>=0;i--){
             while(s.size() > 0 && h[s.top()] > h[i] ){
            l[s.top()] = i+1;
            s.pop();
             }
            s.push(i);
         }
    int ans = INT_MIN;
    for(int i=0;i<n;i++){
    int l_id = (l[i] == -1) ? 0 : l[i];
    int r_id = (r[i] == -1) ? n - 1 : r[i];
        ans = max ( ans , (  h[i]*(r_id - l_id+1)  )   );
    }
    return ans ;
    }
};
```

</details>

### 86. Sliding window maximum

<details>

> The approach is to use a deque to store the indexes of elements in decreasing order. As we slide the window across the array, we ensure two things: first, we remove any elements from the deque that are outside the current window's size; second, we check the back of the deque and remove any indexes whose corresponding values are smaller than the incoming element, as they can't contribute to the maximum value in the current window. To simulate the process, think of it as similar to what we do when finding the Next Greater Element (NGE).

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;  
        vector<int> v;   
        for(int i = 0; i < nums.size(); i++) {
            // Remove elements from the front of the deque that are out of the current window range
            while (dq.size() > 0 && (i - dq.front()) >= k) dq.pop_front();
            // Remove elements from the back of the deque that are smaller than the current element
            while (dq.size() > 0 && nums[i] > nums[dq.back()]) dq.pop_back();
            dq.push_back(i);
            // If the current index 'i' has reached or exceeded the window size 'k - 1'
            // (meaning the window is fully formed), add the maximum element in the current window
            // (which is the front element of the deque) to the result vector 'v'
            if (i >= k - 1)
                v.push_back(nums[dq.front()]);
        }
        return v;  // Return the vector containing maximum values for each sliding window
    }
};
```

</details>

### 87. Implement min stack

<details>

> M1. You can approach this by pushing elements into the stack as usual. The operations are straightforward and use O(2N) space, where the first value is the actual element, and the second value represents the minimum so far. We maintain this second value to efficiently support the `getMin()` function.

> M2. Consider a stack with values 9, 10, 11, 12<-top. Currently, the minimum value is 9. If we encounter an element smaller than 9, that element becomes the new minimum. Let's say the current minimum (`m1`) is 9 and the new minimum (`m2`) is 3. We store a special value at the top of the stack: `2*m2 - m1`. This value ensures that we can safely retrieve the previous minimum (`m1`). Additionally, `2*m2 - m1` will always be less than `m2`, indicating a breakpoint and signaling that it's time to revert to the previous minimum (`m1`). Here's the proof: `m2 < m1` implies `m2 - m1 < 0`, and thus `2*m2 - m1 < m2`.

```cpp
class MinStack {
public:
    MinStack() { }
    stack<long long int> s;
    long long int mi ;
    void push(long long int val) {
        if(s.size()==0) {
            s.push(val);
            mi=val;
            return;
        }
        if(mi <= val){
            s.push(val);
        }else{
            s.push(2*val-mi);
            mi=val;
        }
    }
    void pop() {
        if(  mi > s.top() )  mi = 2*mi-s.top();
        s.pop();
    }
    long long int top() {
            if(   mi <= s.top())  return s.top();
            return mi;
    }
    long long int getMin() {
          return mi;      
    }
};
```

</details>

### 88. Rotten oranges 

<details>

> You can solve this problem using a multi-source BFS approach. Start by placing all the initially rotten oranges with a timestamp of 0 in a queue. Then, use the BFS technique to explore all four possible directions from each rotten orange. If you encounter a fresh orange, mark it as rotten and insert it into the queue with a timestamp incremented by 1 from the parent rotten orange. Keep track of the maximum timestamp encountered during the process. At the end, ensure that all oranges have been rotten, before even returning the max timestamp.

> Please learn how we travel in 4 directions using `dx` and `dy`

```cpp
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        queue<pair<pair<int,int>, int>> q;
        for(int i = 0 ; i<grid.size() ; i++){
            for(int j=0;j<grid[0].size() ; j++){
                if(grid[i][j]==2){
                    q.push( {   { i,j} , 0  } );
                }
            }
        }
    vector<int> dx = {0, 0, -1, 1};  // Corresponding to up, down, left, right
    vector<int> dy = {-1, 1, 0, 0};  // Corresponding to up, down, left, right
    int ma = INT_MIN;
    while(q.size()!=0){
        auto temp =  q.front();
        q.pop();
        int x = temp.first.first;
        int y = temp.first.second;
        int t = temp.second;
        ma = max(ma , t);
        for(int k=0;k<4;k++){
            int nx = x +dx[k];
            int ny = y + dy[k];
            if (   nx>=0 && nx<m && ny>=0 && ny<n && grid[nx][ny] == 1     ){
                  grid[nx][ny] = 2; //make it rotten 
                  q.push( {   { nx, ny } , t+1  } );
            } 
        } 
    }
   for(int i = 0 ; i<grid.size() ; i++) for(int j=0;j<grid[0].size() ; j++) if(grid[i][j]==1) return -1;      
   return ma == INT_MIN ? 0 : ma ;
    }
};
```

</details>

### 89. Stock span problem 

<details>

> Hint: (NGE pattern when you stand at some index named `id`) maximum consecutive days for which stock price was less than or equal to my current day. Dry run with [100 80 60 70 60 75 85] ans: [1 1 1 2 1 4 6]

> This is a somewhat tricky problem. The approach involves maintaining a `vector<int> v`, where you use an `id` to iterate through the upcoming prices. At any point, `v[id]` needs to be calculated. If the upcoming price is less than the price at the top of the stack, you simply push the `(price, id)` pair onto the stack. However, if the upcoming price is higher than the price on top of the stack, we follow the NGE pattern. Since the stack maintains pairs with their respective `id`, this allows you to add the span of days to your `v[id]`. You continue this process until the condition is no longer met. Ultimately, `v[id]` will store the exact answer. This approach resembles dynamic programming because each state `v[i]` represents the span of days to its left.

> [video](https://www.youtube.com/watch?v=eay-zoSRkVc&t=714s)

```cpp
class StockSpanner {
public:
    int id  ;
    vector < int > v ;
    StockSpanner() {
        id = -1;
        v = vector<int>(10000, 1);
    }
    stack< pair <int,int> > s ;
    
    int next(int p) {
        id++;
        if(s.size()==0){
            s.push( {p , id});
            return 1;
        }
        if( p  < s.top().first ){
            s.push( { p, id } );
            return 1;
        }else{
            while( s.size()>0 && s.top().first <= p){
                v[id]+= v[s.top().second]; //DP
                s.pop();
            }
            s.push( { p, id } );
            return v[id];
        }
    }
};
```

</details>

### 90. Maximum of minimum in every window size (hard)

<details>

> A tricky question: maintain the index of NSE for both the left and right directions for any index `i`. Note that by default, we store the `n` index for the right array and `-1` for the left array in case an NSE doesn't exist. Next, calculate `num[i] = abs(l[i] - r[i]) - 1`, which indicates that `h[i]` is the minimum in a window size less than or equal to `num[i]`. However, we must ensure that for window size `num[i]`, we pick the max `h[i]`, which is actually stored in `ans[num[i] - 1]` (with `-1` for 0 indexing). Finally, iterate through the `ans[]` array from the back and update it if a better optimal max answer exists.

> <img width="810" alt="Screenshot 2024-08-16 at 14 49 43" src="https://github.com/user-attachments/assets/58607bc5-d3b8-40c9-8138-e1cd696611dc">


```cpp
vector<int> maxMinWindow(vector<int> h, int n) {
        vector<int> r(n,n);
         vector<int> l(n,-1);
         vector<int> ans(n, INT_MIN);
          vector<int> num(n);
         stack<int> s;
         for(int i=0;i<n;i++){
             while(s.size() > 0 && h[s.top()] > h[i] ){
            r[s.top()] = i;
            s.pop();
             }
            s.push(i);
         }
         while(s.size()!=0) s.pop();
         for(int i=n-1;i>=0;i--){
             while(s.size() > 0 && h[s.top()] > h[i] ){
            l[s.top()] = i;
            s.pop();
             }
            s.push(i);
         }
         for(int i=0;i<n;i++){
             num[i] =   abs( l[i]-r[i] )-1;
             ans[  num[i] -1  ] =    max (h[i] ,     ans[  num[i] -1  ] ) ;
         }
         int ma = ans[n-1];
         for(int i=n-2;i>=0;i--){
             if(ans[i]==INT_MIN)   ans[i] =ma;
             else {
             if (ans[i] < ma) ans[i] = ma;
             else ma = ans[i];
             }
         }
return ans;
}
```

</details>

### 91. Celebrity Problem

<details>

> Push all candidates into the stack. While the stack size is greater than or equal to 2, pick the top 2 elements. Depending on the condition, either one of them will be pushed back, or both will be removed (this is a clear observation). Finally, if you're left with one candidate, cross-check their potential candidate property once more before returning. If the stack is empty, return none.

> <img width="927" alt="Screenshot 2024-08-16 at 15 11 43" src="https://github.com/user-attachments/assets/4c6ad999-a11a-4314-8c30-5368ee17e4da">

```cpp
int findCelebrity(int n) {
stack<int> s ;
	 for(int i=0;i<n;i++) s.push(i);
	 while(s.size() >= 2 ) {
		 int a = s.top();
		 s.pop();
		 int b = s.top();
		 s.pop();
		 if(  knows(a,b) && !knows(b,a) ) s.push(b); 
		 if(  knows(b,a) && !knows(a,b) ) s.push(a); 
	 }
	if (s.size() == 1) {
        int potentialCelebrity = s.top();
        s.pop();
        for (int i = 0; i < n; i++) {
            if (i != potentialCelebrity && (knows(potentialCelebrity, i) || !knows(i, potentialCelebrity))) return -1;  
        }
        return potentialCelebrity;  
    }
    return -1;  
}
```

</details>

### 92. Reverse words in a string 

<details>

> Observe how stringstream is effectively used to remove spaces and extract words from a sentence, regardless of any trailing or leading spaces, no matter how many times they appear. The rest of the code is straightforward.

```cpp
class Solution {
public:
    string reverseWords(string s) {
    stringstream iss(s), oss;
    bool firstWord = true;
    string word;
    while (iss >> word) {
        if (!firstWord) {
            oss << ' ';
        }
        oss << word;
        firstWord = false;
    }
   s= oss.str(); // whatever we did till here, was to show the functionality of stringstream 

        vector<string> v;
        string t = "";
        int f= 1;
        for(int i=0;i<s.size();i++){
            if(s[i]==' '){
                v.push_back(t);
                t="";
            }else{
                t+=s[i];
            }
        }
         v.push_back(t);

        reverse(v.begin(),v.end());
        string an ="";
        for(string s: v){
            an+=s;
            an+=" ";
        }
        an = an.substr(0 , an.size()-1);
        return an;

    }
};
```

> The above solution uses an extra space `O(n)`, a better way is to iterate from the left and keep on breaking the words, and cleverly adding it in this fashion: `ans = temp + " " + ans;`, where `temp` is my current word which we are breaking from L -> R traversal.

```cpp
string result(string s)
{
    int left = 0;
    int right = s.length()-1;
    string temp="";
    string ans="";
    while (left <= right) {
        char ch= s[left];
        if (ch != ' ') {
            temp += ch;
        } else if (ch == ' ') {
            if (ans!="") ans = temp + " " + ans; // case when first-word was already appended in the last 
            else ans = temp;
            temp = "";
        }
        left++;
    }
    
    //If not empty string then add to the result(Last word is added)
    if (temp!="") {
        if (ans!="") ans = temp + " " + ans; 
        else ans = temp; // case when whole string is just a one word 
    }
    
    return ans;    
}
```

</details>

### 93. Longest palindrome in a string (LPS)

<details>

> The idea is quite straightforward: we use an upper diagonal matrix `dp[][]`, where `i` and `j` represent a substring at any given point. Notice how this time I am iterating through my `dp` table with the `j` pointer followed by `i`. This is because, to fill `dp[i][j]`, we need `dp[i+1][j-1]` to be calculated beforehand, which can be achieved by looping in this order. The intuition is clear: if `"nitin"` is a palindrome, then `"_nitin_"` will also be a palindrome if `s[i] == s[j]`. In the end, we search for `1` in our `dp` table and select the maximum length substring.

> <img width="396" alt="Screenshot 2024-08-17 at 13 33 37" src="https://github.com/user-attachments/assets/022a738c-8ee6-4526-951b-88418c93cffb">


```cpp
class Solution {
public:
    string longestPalindrome(string s) {    
        int n = s.size();
        vector<  vector<int> > v ( n , vector<int> ( n ,  0) );
       for (int i = 0; i < n; i++) v[i][i] = 1;
       for(int j=0;j<n;j++){
             for(int i=0;i<n;i++){
                 if (i >= j)  continue;
                 else if (s[i] == s[j] && ( j - i <= 2 || v[i + 1][j - 1])) v[i][j] = 1; 
            }
        }
    int r = -1 ;
    int c = -1;
    int ma = INT_MIN;
    for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (v[i][j] && abs(i - j) + 1 > ma) {
                    r = i;
                    c = j;
                    ma = abs(i - j) + 1;
                }
        }
     }
return s.substr( r , c-r+1);
}
};
```

</details>

### 94. Roman number to integer and vice-versa

<details>

> The idea is to iterate from the end of the string towards the beginning, setting up `prevValue` and `result` variables. As you move through each position, if the current value is greater than or equal to the `prevValue` value, it's safe to add the Roman numeral's integer value to the result. Otherwise, subtract the current value from the result. Try this with examples like "III" and "IV" to quickly understand the concept.

```cpp
class Solution {
public:
    int romanToInt(string s) {
        unordered_map<char, int> romanValues = {
            {'I', 1}, {'V', 5}, {'X', 10},
            {'L', 50}, {'C', 100}, {'D', 500},
            {'M', 1000}
        };
        int result = 0;
        int prevValue = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            int currentValue = romanValues[s[i]];
            if (currentValue >= prevValue) result += currentValue;
             else result -= currentValue;
            prevValue = currentValue;
        }
        
        return result;
    }
};
```

</details>

### 94. Implement ATOI/STRSTR (string-to-integer)

<details>


```cpp
class Solution {
public:
    int myAtoi(string s) {
        int i = 0;
        int n = s.length();
        long long result = 0;  // Use long long to handle potential overflow
        int sign = 1;
        // Step 1: Read and ignore leading whitespace
        while (i < n && s[i] == ' ') i++;
        // Step 2: Check for '+' or '-'
        if (i < n && (s[i] == '+' || s[i] == '-')) {
            sign = (s[i] == '-') ? -1 : 1;
            i++;
        }
        // Step 3: Read and convert digits
        while (i < n && isdigit(s[i])) {
            result = result * 10 + (s[i] - '0');
            // Step 4: Check for overflow and clamp if necessary
            if (result * sign < INT_MIN) return INT_MIN;
            if (result * sign > INT_MAX) return INT_MAX;
            i++;
        }
        // Step 5: Return the integer as the final result
        return static_cast<int>(result * sign);
    }
};
```

</details>

### 95. Longest common prefix

<details>

> Try finding the shortest string, and then apply binary search over its length to check whether the prefix created with it matches exactly with all the other strings.

```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (strs.empty()) return ""; 
        string shortestStr = strs[0];
        for (const string& str : strs) if (str.length() < shortestStr.length()) shortestStr = str;
        // Binary search approach to find the longest common prefix.
        int left = 0;
        int right = shortestStr.length() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            string prefix = shortestStr.substr(0, mid + 1);
            bool isCommonPrefix = true;
            for (const string& str : strs) {
                if (str.substr(0, mid + 1) != prefix) {
                    isCommonPrefix = false;
                    break;
                }
            }
            if (isCommonPrefix) left = mid + 1; // Move to the right half.
 	    else right = mid - 1; // Move to the left half.
        }
        return shortestStr.substr(0, left);
    }
};
```

</details>

### 96. Rabin Karp 

<details>

> Rabin-Karp works by first calculating the hash of the pattern, then iterating through the text string to find if the pattern's hash appears at any index `i` to `j`, where `j - i + 1 = pattern.size()`. If a match in the hash is found, we double-check by iterating from `i` to `j` and comparing each character in the substring with the pattern. The hash is calculated using:

> `char_code * pow(base, expo)`, where `base` is generally the total number of characters, and `expo` ranges from `0, 1, 2,...` up to `pattern.size() - 1`, moving from right to left.

> We introduce a modular exponentiation function to avoid integer overflow. As we iterate through the text string, we cleverly calculate the hash of each substring of length `pattern.size()` using the previous hash value, which reduces the time complexity to linear, `O(n)`.

```cpp
long long int mod = 1000000000 + 7;
long long int mod_pow(long long int base, int exponent) {
    long long int result = 1;
    while (exponent > 0) {
        if (exponent & 1) // equivalent to (exponent % 2 == 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exponent >>= 1; // equivalent to (exponent /= 2)
    }
    return result;
}
vector<int> stringMatch(string t, string p) {
    vector<int> v;
    long long int hash = 0; // used to store for pattern 
    int base = 26;
    int id = 0;
    for (int i = p.size() - 1; i >= 0; i--) {
        int x = p[i] - 'a';
        hash = (hash + (x * mod_pow(base, id))) % mod;
        id++;
    }
    id = 0;
    if (p.size() <= t.size()) {
        id = 0;
        long long int hash1 = 0; //used to store the first window hash in text, and then we cleverly roll the next window hash  
        for (int i = p.size() - 1; i >= 0; i--) {
            int x = t[i] - 'a';
            hash1 = (hash1 + (x * mod_pow(base, id))) % mod;
            id++;
        }
        id = 0;
        while (id <= t.size() - p.size()) {
            if (id != 0) { // because for id = 0 we have already calculated the hash value in `hash1`
                hash1 = (hash1 * base) % mod;
                hash1 = (hash1 - ((t[id - 1] - 'a') * mod_pow(base, p.size()))) % mod;
                hash1 = (hash1 + ((t[id + p.size() - 1] - 'a') * mod_pow(base, 0))) % mod;
                if (hash1 < 0) hash1 = (hash1 + mod) % mod; // Ensure the result is positive   
            }
            if (hash1 == hash) {
                bool match = true;
                for (int i = 0; i < p.size(); i++) { // still we can't fully trust on hash so we need to check every character one by one
                    if (t[id + i] != p[i]) {match = false;break;}
                }
                if (match) v.push_back(id + 1);
            }
            id++;
        }
    } else return {}; //pattern is greater than text given
    return v;
}
```

> Note that this question is essentially an implementation or an extended form of the above Rabin-Karp case. Here, we return the first index where the hash matches and use simple math to determine the minimum number of times you should repeat string `a` so that string `b` becomes a substring of it. If it is impossible for `b` to be a substring of `a` after repeating it, return `-1`. We try repeating string `a` at least twice but before that repeat it until the size of `t` is `<= p.size()`.

```cpp
int repeatedStringMatch(string a, string b) {
        string t = "";
        string p = "";
        // a (repeat) and we need to search for b in a 
        p = b ;
        while(   t.size() <= 2*p.size()    ) t+= a ; 
        t+=a; // to get why we need to think that there can be matches at the junction point
        int id = stringMatch(t , p ) ;
        if(id == - 1) return -1 ;
        id = id + p.size(); //pointing to the end of the pattern match in the text string, since it helps you to know in which repeated region of string `a` you are currently pointing to
        if( id % a.size() == 0 ) return id/a.size() ;
        else return id/a.size() +1 ;
    }
};
```

</details>

### 97. Z function  

<details>

> <img width="637" alt="Screenshot 2024-08-17 at 17 08 19" src="https://github.com/user-attachments/assets/55b6321a-1171-42a4-b3a8-6ea1df923e1f">

> `"aabxaabxcaabxaabxaay"` dry run it, and understand the intuition yourself

> <img width="677" alt="Screenshot 2024-08-17 at 18 34 33" src="https://github.com/user-attachments/assets/284841ef-cb37-48d6-9d68-cde6953fe3c7">


```cpp
vector<int> z_function(string s) {
    int n = s.size();
    vector<int> z(n, 0);
    int l = 0, r = 0;
    for(int i = 1; i < n; i++) {
        if(i < r) z[i] = min(r - i, z[i - l]); // a high optimization done here to use stored answer and do a good skipping part, intuition is that if z[id] = x means we have a length of substring x at index id which exactly matches with substring starting at index =0, now if we are in a situation when my current index is `i` which is lesser than r that means we can take minimum of `r-i` that shows how many chars we have towards right = [i,r) or `z[i-l]` which is my previously stored answer, we can use it because instead of comparing the substring from index = `i` char by char with index = 0, we can use it because for index `i-l` we have done the same thing previously so just use it that's what DP is all about.

        while(i + z[i] < n && s[z[i]] == s[i + z[i]])  z[i]++; // It compares the upcoming characters of my string starting with index `0` and `i`, please notice we update z[i] that help us to quickly jump over the indices of my both substring one starting from index 0 while other starting from index i.

        if(i + z[i] > r) { // we try to reset the l to r box size
            l = i;
            r = i + z[i];
        }
    }
    return z;
}
```

> use case, find string `needle` in `haystack`: O(n) cost

```cpp
 int strStr(string haystack, string needle) {
        if (needle.empty()) return 0;
        string concat = needle + "#" + haystack;
        vector<int> z_values = z_function(concat);
        int needle_length = needle.length();
        for (int i = needle_length + 1; i < concat.length(); i++) {
            if (z_values[i] == needle_length) {
                return i - needle_length - 1;
            }
        }
        return -1; // Needle not found in haystack
    }
```

</details>

### 98. KMP algo/LSP(pi) array

<details>

> The value of `pi[i]` represents the length of the longest proper prefix which is also a suffix. We initialize `pi[0] = 0` and start filling the array from index 1. To determine `pi[i]`, we use the value of `pi[i-1]`, which gives the exact previous index of the string that needs to be compared with the `i`th index.

```
Let's clear it: 
i :         0   1   2    3   4    5   6   7   8
s[i]:       a   a   b    a   a    b   a   a   a
p[i]:       0   1   0    1   2
```

> Intuition 1: Imagine you want to fill `p[5]`. First, store `p[4]` in `j`, which is 2. This value tells us that we have a proper prefix of length 2, which is also a suffix. Naturally, the next character to compare will be at index 2, so we check if `s[2] == s[5]`. If they match, increment `j` and store it in `pi[5]`. The same logic applies for filling `pi[6]` and `pi[7]`.

```
i :         0   1   2    3   4    5   6   7   8
s[i]:       a   a   b    a   a    b   a   a   a
p[i]:       0   1   0    1   2    3   4   5
```

> Intuition 2: If `s[i] != s[j]`, for example, `s[8] != s[5]`, we backtrack to index 5. At index 5, we know it was calculated using index 4, where `pi[4]` is 2. This tells us to compare `s[8]` with `s[2]`. If they are not equal, we continue backtracking to index 2. Index 2 was determined using index 1, where `pi[1]` is 1, indicating a longest proper prefix-suffix (LPS) of length 1. Now, compare `s[8]` with `s[1]`. If `s[1] == s[8]`, then we increment the value at that index by 1.

```
i :         0   1   2    3   4    5   6   7   8
s[i]:       a   a   b    a   a    b   a   a   a
p[i]:       0   1   0    1   2    3   4   5   2
```

```cpp
vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi;
}
```

</details>

### 99. Minimum characters to be inserted in front to make it palindromic

<details>

> Refer Q.98, then use that function to solve this 

```cpp
int Solution::solve(string A) {
    string combined = A + "#" + string(A.rbegin(), A.rend()); // Concatenate A and its reverse with a unique separator "#"
    vector<int> pi = prefix_function(combined); // Calculate the prefix function for the combined string
    int maxPalindromicLength = pi.back(); // The length of the longest palindromic suffix
    // The minimum characters needed to make A palindromic is (A.length() - maxPalindromicLength)
    return A.length() - maxPalindromicLength;
}
```

</details>

### 100. Anagrams

<details>


```cpp
bool isAnagram(string s1, string s2) {
    if (s1.length() != s2.length()) {
        return false;
    }
    vector<int> freq(26, 0);
    for (char c : s1) {
        freq[c - 'a']++;
    }
    for (char c : s2) {
        freq[c - 'a']--;
    }
    for (int count : freq) {
        if (count != 0) {
            return false;
        }
    }
    return true;
}
```

</details>

### 101. Count and say

<details>
	
> <img width="159" alt="Screenshot 2024-08-17 at 16 59 41" src="https://github.com/user-attachments/assets/7219dd37-854d-4a35-ae23-fbd207c8437b">


```cpp
class Solution {
public:
    string countAndSay(int n) {
        if (n <= 0) return ""; 
        string result = "1";
        for (int i = 1; i < n; i++) {
            string next;
            int count = 1;
            for (int j = 0; j < result.length(); j++) {
                if (j + 1 < result.length() && result[j] == result[j + 1]) count++;
                else {
                    next += to_string(count) + result[j];
                    count = 1;
                }
            }
            result = next;
        }
        return result;
    }
};
```

</details>

### 102. Compare Version numbers   

<details>

> Please see the use of the `getline` function

```cpp
class Solution {
private:
    vector<int> splitVersion(string version) {
        vector<int> result;
        stringstream ss(version);
        string part;
        while (getline(ss, part, '.')) result.push_back(stoi(part));
        return result;
    }
};
public:
    int compareVersion(string version1, string version2) {
        // Split the version strings by the dot ('.') character
        vector<int> v1 = splitVersion(version1);
        vector<int> v2 = splitVersion(version2);
        int n = max(v1.size(), v2.size());
        for (int i = 0; i < n; i++) {
            int part1 = (i < v1.size()) ? v1[i] : 0;
            int part2 = (i < v2.size()) ? v2[i] : 0;
            if (part1 < part2) return -1;
            else if (part1 > part2) return 1;
            
        }
        return 0;
    }
```

</details>

### 103. Inorder 

<details>


```cpp
class Solution {
public:
    void f (TreeNode* r,  vector<int>& v){
        if(!r) return;
        f(r->left, v);
        v.push_back(r->val);
        f(r->right, v);
    }
    vector<int> inorderTraversal(TreeNode* root) {
         vector<int> v;
         f(root, v);
         return v;
    }
};
```

</details>

### 104. Preorder 

<details>

```cpp
class Solution {
public:
    void preorderHelper(TreeNode* r, vector<int>& v) {
        if (!r) return;
        v.push_back(r->val);    // Visit the root node
        preorderHelper(r->left, v);  // Traverse left subtree
        preorderHelper(r->right, v); // Traverse right subtree
    }
    
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> v;
        preorderHelper(root, v);
        return v;
    }
};
```

</details>

### 105. Postorder

<details>

```cpp
class Solution {
public:
    void postorderHelper(TreeNode* r, vector<int>& v) {
        if (!r) return;
        postorderHelper(r->left, v);  // Traverse left subtree
        postorderHelper(r->right, v); // Traverse right subtree
        v.push_back(r->val);    // Visit the root node
    }
    
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> v;
        postorderHelper(root, v);
        return v;
    }
};
```

</details>

### 106. Morris Inorder 

<details>

> The intuition here stems from the fact that since recursion or a stack isn't allowed, there must be a way to return to the current root node. Imagine yourself at a node with a left and right subtree. For pre/inorder traversal, you know that the traversal follows a left-to-right fashion. Consider a tree with nodes like 1, 2, 3, n, 4, 5, n, n, n, n, n, 6 (refer to a tree diagram for clarity). When you're at node 1, there must be a thread that connects node 6 back to node 1. Similarly, when you're at node 2, a thread must connect node 4 back to node 6.

> These threads help us revisit a node and allow us to move to the other half of the tree. This is achieved by checking if the last node in the subtree points back to the current node. A simple observation is that if the current node doesn't have a left subtree, you can just push the value and move to the right subtree.

> The logic for preorder and inorder traversals is nearly identical, differing by just a single line of code. However, postorder traversal is a bit different but still uses similar code and ultimately prints the result in reverse order.

```cpp
```

</details>

### 107. Morris Preorder 

<details>


```cpp
```

</details>

### 108. Left view of BT

<details>

> The intuition is to simply traverse the tree left and then right. Now, imagine the tree as having levels: lvl0, lvl1, and so on. Clearly, when you enter the recursion for the first time using DFS with a unique `lvl` value, the `r` node always reflects the leftmost value of that level, which can be seen from the left side.

```cpp
void f(Node* r, std::vector<int>& l, int lvl){
    if (!r) return;
    if (l.size() == lvl) l.push_back(r->data);
    f(r->left, l, lvl + 1);
    f(r->right, l, lvl + 1);
}

vector<int> leftView(Node* r) {
    std::vector<int> l;
    f(r, l, 0);
    return l;
}
```

</details>

### 109. Bottom view of BT

<details>

> Refer Q.110

> The condition `else if (hd[dist].second <= depth)` ensures that if two nodes are at the same horizontal distance, the node that is at a greater depth (i.e., the node appearing later in the level-order traversal) will be stored. This is because when two nodes are at the same level but different subtrees (left and right), the right subtree node is processed after the left subtree node, so the right subtree node will overwrite the left subtree node. Thus, the node that appears at the bottom level at each horizontal distance will be retained.

```cpp
class Solution {
public:
    void f(Node *r, map<int, pair<int, int>>& hd, int dist, int depth) {
        if (!r) return;
        if (hd.find(dist) == hd.end())  hd[dist] = {r->data, depth};
        else if (hd[dist].second <= depth) hd[dist] = {r->data, depth}; 
        f(r->left, hd, dist - 1, depth + 1);
        f(r->right, hd, dist + 1, depth + 1); 
    }

    vector<int> bottomView(Node *r) {
        map<int, pair<int, int>> hd; // horizontal distance
        f(r, hd, 0, 0);
        vector<int> an;
        for (auto it = hd.begin(); it != hd.end(); ++it) an.push_back(it->second.first);
        return an;
    }
};
```

</details>

### 110. Top view of BT

<details>

> There are two tricks involved. First, when performing DFS, you must ensure that if two or more nodes lie on the same horizontal distance (HD) from the root node, the one closest to the root should take the position. If both are at the same depth, the leftmost node should get the place. This is ensured by using the depth variable and calling the recursion for the left node first, followed by the right node.

```cpp
class Solution {
public:
    void f(Node *r, map<int, pair<int, int>>& hd, int dist, int depth) {
        if (!r) return;
        if (hd.find(dist) == hd.end())  hd[dist] = {r->data, depth};
        else if (hd[dist].second > depth) hd[dist] = {r->data, depth};
        f(r->left, hd, dist - 1, depth + 1);
        f(r->right, hd, dist + 1, depth + 1); 
    }

    vector<int> topView(Node *r) {
        map<int, pair<int, int>> hd; // horizontal distance
        f(r, hd, 0, 0);
        vector<int> an;
        for (auto it = hd.begin(); it != hd.end(); ++it)    an.push_back(it->second.first);
        return an;
    }
};
```

</details>

### 111. Preorder Postorder Inorder in single Traversal 

<details>

```cpp
void f (TreeNode* r,  vector<int>& in , vector<int>& pre , vector<int>& post ){
        if(!r) return;
        pre.push_back(r->data);
        f(r->left, in,pre,post);
        in.push_back(r->data);
        f(r->right, in,pre,post);
        post.push_back(r->data);
    }
vector<vector<int>> getTreeTraversal(TreeNode *r){
    vector<vector<int>> an ;
    vector<int> in ;
    vector<int> pre ;
    vector<int> post ;
    f(r,in,pre,post);
    an.push_back(in);
        an.push_back(pre);
            an.push_back(post);
            return an;
}
```

</details>

### 112. Vertical Order Traversal

<details>

> The concept is straightforward: use an appropriate data structure, `map<int, vector<pair<int, int>>> hd`, where the key represents the horizontal distance (HD) either to the left or right, and each node's value is stored along with its depth. This allows for sorting later when we need to create the column vector for a particular horizontal distance.

```cpp
class Solution {
public:
    void f(TreeNode *r, map<int, vector<pair<int, int>>>& hd, int dist, int depth) {
        if (!r) return;
        hd[dist].push_back({depth, r->val}); // Store depth along with the node value
        f(r->left, hd, dist - 1, depth + 1);
        f(r->right, hd, dist + 1, depth + 1); 
    }
    vector<vector<int>> verticalTraversal(TreeNode *r) {
        map<int, vector<pair<int, int>>> hd; 
        f(r, hd, 0, 0);
        vector<vector<int>> an;
        for (auto& outer_pair : hd) {
            vector<pair<int, int>>& nodes = outer_pair.second;
            sort(nodes.begin(), nodes.end()); // Sort nodes by depth
            vector<int> column;
            for (auto& node : nodes) 
                column.push_back(node.second); // Add the node values to the column
            an.push_back(column);
        }
        return an;
    }
};
```

</details>

### 113. Root to all Leaf Paths in BT

<details>

> The intuition is based on identifying whether a node is a leaf. If it is a leaf, consider `cur` in your answer. To determine whether a node is a leaf or not, you follow the steps shown below.

```cpp
bool f (BinaryTreeNode<int> *r , vector <string>& an , vector<int>& cur ){
     if (!r) return 0;
     cur.push_back(r->data);
     bool  lft  = f(r->left , an,cur); // lft tells whether current node have left child or not 
     bool  rght = f(r->right, an, cur); // rght tells whether current node have right child or not
        if( !lft && !rght){ // current node is a leaf node
            string temp = "";
            for(auto el : cur) temp = temp +  to_string(el) + " ";
            an.push_back(temp);
    }
     cur.pop_back();
     return 1; // not a leaf node
}

vector <string> allRootToLeaf(BinaryTreeNode<int> * r) {
   vector <string> an ;
   vector<int> cur ;
   bool chk = f(r,an,cur); //chk is dummy 
   return an; 
}
```

</details>

### 114. Max width of BT

<details>

> Initially, push the root node and calculate the width of level 0. Then, to find the width of level 1, pop the root node and push its left and right children into a temporary queue (`tmp`), which is then copied to the original queue. Note that null nodes are not pushed. This process is repeated until the last level.

> To avoid integer overflow, use `long long int` data type. Additionally, reassign index values for every level, focusing solely on the width. This reassignment is handled when copying `tmp` to the original queue `q`.

> We are doing level order traversal, tbh, but a little bit in a modified way.

```cpp
class Solution {
public:
    int widthOfBinaryTree(TreeNode* r) {
        long long int maxi = 0; // Use long long int for the maximum width
        queue<pair<TreeNode*, long long int>> q;
        q.push({ r , 0 });
        while (!q.empty()) {
            long long int id1 = q.front().second;
            long long int id2 = q.back().second;
            maxi = max(maxi, id2 - id1 + 1);
            queue<pair<TreeNode*, long long int>> tmp;
            long long int cur_size = q.size();
            
            for (long long int i = 0; i < cur_size; i++) {
                pair<TreeNode*, long long int> cur = q.front();
                if (cur.first->left)  tmp.push({cur.first->left, 2 * cur.second + 1});
                if (cur.first->right) tmp.push({cur.first->right, 2 * cur.second + 2});
                q.pop();
            }
            
            if (!tmp.empty()) { // normalize it to avoid INT overflow
                long long int mini = tmp.front().second;
                while (!tmp.empty()) {
                    q.push({tmp.front().first, tmp.front().second - mini});
                    tmp.pop();
                }
            }
        }

        return static_cast<int>(maxi); // Convert back to int before returning
    }
};

```

</details>

### 115. Level order traversal (L->R)

<details>

> Refer Q.114

```cpp
class Solution {
public:
   vector<vector<int>> levelOrder  (TreeNode* r) {
        if(!r) return {};
        queue<TreeNode*> q;
        q.push(r);
        vector<vector<int>> an;
        while (!q.empty()) {
            queue<TreeNode*> tmp;
            long long int cur_size = q.size();
            vector<int> v;
            for (long long int i = 0; i < cur_size; i++) {
                v.push_back(q.front()->val);
                if(q.front()->left) tmp.push(q.front()->left);
                if(q.front()->right) tmp.push(q.front()->right);
                q.pop();
            }
            an.push_back(v);
            if (!tmp.empty()) {               
                while (!tmp.empty()) {
                    q.push(tmp.front());
                    tmp.pop();
                }
            }
        }
        return an;
    }
};
```

</details>

### 116. Height of BT 

<details>

> An easy way to maintain the height of the current node from the root is to update the depth whenever you enter the recursion. Ensure you always store the maximum possible depth in the `dep` variable using `dep = max(dep, ht)`.

```cpp
class Solution {
public:
     int dep = INT_MIN;
     void f (TreeNode* r, int ht){
         if(!r) return ;
         dep = max(dep , ht);
         f(r->left, ht+1);
         f(r->right, ht+1);

     }
    int maxDepth(TreeNode* r) {
        f(r, 1);
        return dep==INT_MIN?0:dep;
    }
};
```
> To find the height of the tree, we calculate the height of the left and right subtrees and return `1 + max(left_ht, right_ht)`. This approach ensures that when the recursion backtracks to the previous node, it receives the maximum height of both subtrees. The `+1` is added to account for the current node itself, which contributes `+1` height to either the left or right subtree.

```cpp
class Solution {
public:
    int f (TreeNode* r){
        if(!r) return 0;
        int left_ht = f(r->left);
        int rght_ht = f(r->right);
        return 1 + max(left_ht , rght_ht);
    }
    int maxDepth(TreeNode* r) {
       return f(r); 
    }
};
```

</details>

### 117. Diameter of BT 

<details>

> Length of the longest path between any two nodes in a tree

> Now, we maintain a `dia` parameter that holds the maximum possible diameter of the tree at any node. You might wonder why there is no `+1` in the `dia` calculation. The reason is that, for a path like `a - b - c - d`, if `b` is the node we're considering with left height `1` and right height `2`, we store `1 + 2` instead of `1 + 2 + 1`. This is because the path length is counted by the number of edges (i.e., `-`), not by the number of nodes.

```cpp
class Solution {
public:
    int dia = INT_MIN;
    int f (TreeNode* r){
        if(!r) return 0;
        int left_ht = f(r->left);
        int rght_ht = f(r->right);
        dia = max(dia, left_ht+rght_ht);
        return 1 + max(left_ht, rght_ht);
    }
    int diameterOfBinaryTree(TreeNode* r) {
        int pass = f(r);
        return dia; 
    }
};
```

</details>

### 118. BT is Height balanced or not 

<details>

> At any point, if you notice that the height difference between the left and right subtrees is greater than 1, the tree is not balanced. However, we still return `1 + max(left_ht, right_ht)` because we need to provide the height of the binary tree to the previous node when backtracking. This ensures that height calculations can be correctly propagated up the tree. 🌴

```cpp
class Solution {
public:
    bool flag = 1;
    int f (TreeNode* r){
        if(!r) return 0;
        int left_ht = f(r->left);
        int rght_ht = f(r->right);
        if(abs(left_ht-rght_ht) >1) flag = 0;
        if(left_ht>rght_ht)  return 1 + left_ht;
        else   return 1 + rght_ht;
    }
     bool isBalanced(TreeNode* r) {
       int pass = f(r); 
       return flag; 
    }
};
```

</details>

### 119. LCA in BT  

<details>

> One approach is to store the path from the root node to nodes `p` and `q`, and then iterate through both paths to find the first mismatch. The node before this mismatch is the Lowest Common Ancestor (LCA). However, finding the path requires O(N) time complexity and O(N) extra space, which can be cumbersome. 

> To avoid extra space complexity, use DFS to find the LCA. During traversal, propagate the node address as you backtrack. The intuition is to continue traversing left and right. If you reach a node where either left or right is null, return the non-null node. If both are null, return any. While iterating, if you encounter a node whose address matches either `p` or `q`, return that node. If a node has both left and right children non-null, return the current node's address as it is the LCA.

```cpp
class Solution {
public:
    TreeNode* f(TreeNode* r, TreeNode* p, TreeNode* q){
       if(!r) return NULL;
       if(r == p) return p ;
       if(r == q) return q ;
       TreeNode* left  =  f(r->left, p, q );
       TreeNode* right =  f(r->right,p ,q );
       if (left != NULL && right != NULL ) return r ;
       if(left == NULL) return right;
       if(right == NULL) return left; 
       return NULL ;
    }
    TreeNode* lowestCommonAncestor(TreeNode* r, TreeNode* p, TreeNode* q) {
        return f(r,p,q);
    }
};
```

</details>

### 120.  Check if the two trees are identical or not 

<details>


```cpp
class Solution {
public:
    bool f(TreeNode* p, TreeNode* q){
        if(!p) if(!q) return 1; else return 0;
        if(!q) return 0;
        if(p->val!=q->val) return 0;
        return f(p->left , q->left) && f(p->right, q-> right);
    }
    bool isSameTree(TreeNode* p, TreeNode* q) {
        return f(p ,q);
    }
};
```

</details>

### 121. Zig-Zag Traversal of BT

<details>

> Refer [Q.115](https://github.com/kuspia/Striver_SDE-shortnotes-/blob/main/README.md#115-level-order-traversal-l-r)

```cpp
class Solution {
public:
   vector<vector<int>> zigzagLevelOrder (TreeNode* r) {
        if(!r) return {};
        queue<TreeNode*> q;
        q.push(r);
        vector<vector<int>> an;
        bool f = 0 ;
        while (!q.empty()) {
            queue<TreeNode*> tmp;
            long long int cur_size = q.size();
            vector<int> v;   
            f = !f;
            for (long long int i = 0; i < cur_size; i++) {
                v.push_back(q.front()->val);
                if(q.front()->left) tmp.push(q.front()->left);
                if(q.front()->right) tmp.push(q.front()->right); 
                q.pop();
            }
            if(!f) reverse(v.begin(), v.end());
            an.push_back(v);   
            if (!tmp.empty()) {               
                while (!tmp.empty()) {
                    q.push(tmp.front());
                    tmp.pop();
                }
            }
        }
        
        return an;
    }
};
```

</details>

### 122. Boundary Traversal of BT  

<details>


> [Anticlockwise-Iteration] The idea is to start by storing the root node in the answer and then combine it with the vectors for the left subtree, leaf nodes, and right subtree. Finding leaf nodes is straightforward, but for the left and right vectors, it's important to observe that backtracking is not necessary!

> Please refer to the code to understand the approach. Note that the `right` vector must be inserted in reverse order, and the conditional checks are carefully designed to ensure that everything is inserted correctly without errors.

```cpp
int f_leaf(TreeNode<int> *r, vector<int>& leaf, int lvl){
    if (!r) return 0;
    int left_side =  f_leaf(r->left, leaf, lvl + 1);
    int right_side = f_leaf(r->right, leaf, lvl + 1);
    if(!left_side && !right_side)  leaf.push_back(r->data);
    return 1;
}
void f_left(TreeNode<int> *r, vector<int>& left){
    while( r && (r->left != NULL || r->right != NULL) ){ // can never be a leaf node 
        left.push_back(r->data);
        r = r->left ? r->left : r->right ;
    }
}
void f_right(TreeNode<int> *r, vector<int>& right){ // can never be a leaf node 
    while( r && (r->right != NULL || r->left != NULL) ){
        right.push_back(r->data);
        r = r->right ? r->right : r->left ;
    }
}
vector<int> traverseBoundary(TreeNode<int> *r)
{
	if(!r)  return {};
	vector<int> leaf , left, right, answer ;
	answer.push_back(r->data); 
	int pass = f_leaf(r, leaf, 0);
	f_left(r->left, left);
	f_right(r->right, right);
	answer.insert(answer.end(), left.begin(), left.end());
	answer.insert(answer.end(), leaf.begin(), leaf.end());
	answer.insert(answer.end(), right.rbegin(), right.rend());
	return answer ; 
}
```

</details>

### 123. Maximum path sum  

<details>

> This is a very tricky problem. The idea is to return the maximum of the left and right subtree sums plus the current node's value. This approach ensures that each recursion call receives the best possible sum from its subtrees. 

> At any point, we also maintain a `max_sum` to keep track of the overall maximum sum encountered. The key trick is to reset the sum to `0` if either the left or right subtree sum is less than `0`, as including a negative sum would reduce the total sum of the path.

```cpp
class Solution {
public:
    int ma = INT_MIN;
    int f(TreeNode* n)
        {
        if(!n) return 0; 
        int l = f(n->left);
        int r = f(n->right);
        int v = n->val;
        if(l<0) l = 0 ; 
        if(r<0) r = 0; 
        ma = max(ma, v+l+r);
        return max(l,r) + v;
        }
    int maxPathSum(TreeNode* root) {
        int pass = f(root);
        return ma==INT_MIN ? 0:ma;  
    }
};
```

</details>

### 124. BT from Inorder and Preorder  

<details>

> Refer Q.125

```cpp
class Solution {
public:
    TreeNode* f(vector<int>& in, vector<int>& pre, map<int, int>& mp, int preStart, int preEnd, int inStart, int inEnd) {
        if (preStart > preEnd || inStart > inEnd)  return NULL;
        int rootValue = pre[preStart];
        TreeNode* root = new TreeNode(rootValue);
        int rootIndexInInorder = mp[rootValue];
        int numRightNodes = inEnd - rootIndexInInorder;
        int numLeftNodes = rootIndexInInorder - inStart;
root->left = f(in, pre, mp, preStart+1, preStart + numLeftNodes, inStart, rootIndexInInorder - 1);
root->right = f(in, pre, mp, preEnd - numRightNodes + 1, preEnd, rootIndexInInorder + 1, inEnd);
    return root;
    }
    TreeNode* buildTree(vector<int>& pre, vector<int>& in) {
        map<int, int> mp;
        int n = in.size();
        for (int i = 0; i < n; i++) mp[in[i]] = i;
        return f(in, pre, mp, 0, n - 1, 0, n - 1);
    }
};
```

</details>

### 125. BT from Inorder and Postorder  

<details>

> The idea is to maintain four pointers: `start` and `end` for both the `postorder` and `inorder` vectors, and a map to find the index of any `postorder` element in the `inorder` vector. This helps split the tree into left and right subtrees. Let: `inorder: 9 3 15 20 7` and `postorder: 9 15 7 20 3`

> The root of the tree is `3`, which is at position `2` in the `inorder` vector. This tells us that `9` forms the left subtree, and `15 20 7` forms the right subtree.

> For the right subtree: `inorder: 15 20 7` and `postorder: 15 7 20`
  
> `if (postStart > postEnd || inStart > inEnd)` indicates that the recursion should return `null` for that subtree.

```cpp
class Solution {
public:
    TreeNode* f(vector<int>& in, vector<int>& post, map<int, int>& mp, int postStart, int postEnd, int inStart, int inEnd) {
        if (postStart > postEnd || inStart > inEnd)  return nullptr;
        int rootValue = post[postEnd];
        TreeNode* root = new TreeNode(rootValue);
        int rootIndexInInorder = mp[rootValue];
        int numRightNodes = inEnd - rootIndexInInorder;
root->left = f(in, post, mp, postStart, postEnd - numRightNodes - 1, inStart, rootIndexInInorder - 1);
root->right = f(in, post, mp, postEnd - numRightNodes, postEnd - 1, rootIndexInInorder + 1, inEnd);
        return root;
    }
    TreeNode* buildTree(vector<int>& in, vector<int>& post) {
        map<int, int> mp;
        int n = in.size();
        for (int i = 0; i < n; i++) mp[in[i]] = i;
        return f(in, post, mp, 0, n - 1, 0, n - 1);
    }
};
```

</details>

### 126. Symmetric BT 

<details>

> Assume that your tree is symmetric. Take two iterators that traverse the tree in opposite directions. If the tree is symmetric over the central mirror axis, you will observe that each recursive call will iterate over nodes with the same value. However, if you encounter a mismatch—such as one iterator reaching `NULL` while the other is still non-null, or if the values at the nodes pointed to by the iterators differ—return 0.

```cpp
class Solution {
public:
    bool f(TreeNode* n1, TreeNode* n2){
  if(!n1 && !n2) return 1;;
 if(!n1 || !n2) return 0;
if (n1->val != n2-> val) return 0;
       return f(n1->left , n2->right) && f(n1->right,n2->left);
     }
    bool isSymmetric(TreeNode* root) {
    return f(root,root);   
    }
};
```

</details>

### 127. Flatten BT to LL 

<details>

> The question is very tricky, so what we do is iterate through the tree. If we encounter a `null`, nothing happens. In the case of a leaf node, it is returned to the previous recursion call. The only way is to do a dry-run and get the gist of how it works.

> <img width="600" alt="Screenshot 2024-08-18 at 18 53 58" src="https://github.com/user-attachments/assets/8a86ade8-554a-4955-b974-7d3197542d4b">

```cpp
class Solution {
public:
    TreeNode* f(TreeNode* n){
    if(!n) return NULL;
    TreeNode* l = f(n->left);
        TreeNode* t = n->right; // we store the right subtree in a `t` variable (which we will process recursively and flatten)
        if(l){
            n->right = l;
            n->left = NULL;
        }
        TreeNode* r = f(t); // At this point for the nth node right sub-tree is well flattened which we need to connect at l->right->right ...... last node 
        if(r && l)  {       
	while (l->right)  l = l->right;
	l->right = r;
	}
        return n; // return nth node address to the guy whosoever called it 
    }
    void flatten(TreeNode* r) {
        TreeNode* pass = f(r);
    }
};
```

</details>

### 128. Convert a binary tree into its mirror tree.

<details>

> Before doing this once check out [Q.116](https://github.com/kuspia/Striver_SDE-shortnotes-/tree/main#116--check-if-the-two-trees-are-identical-or-not) and [Q.122](https://github.com/kuspia/Striver_SDE-shortnotes-/tree/main#122-symmetric-bt)

```cpp
class Solution {
  public:
    // Function to convert a binary tree into its mirror tree.
   Node* f(Node* n){
       if(!n) return NULL;
    Node* l = f(n->left);
    Node* r = f(n->right);
    n->left =r;
    n->right=l;
    return n;
   }
    void mirror(Node* n) {
        Node* pass = f(n);
    }
};
```

</details>

### 129. Check for children sum property

<details>

>Observe that if at any point `n` is `null`, it indicates a null node, so you should return 1. For any leaf node, the value will always be 1, which is ensured by the conditional check `if(val2 != 0 ...)`. This check is specifically designed to handle cases where `val2` is non-zero, ensuring that the function correctly identifies and processes leaf nodes. The condition is written this way to ensure that non-null and valid leaf nodes are properly accounted for in the function's logic.

```cpp
bool f (Node* n){
    if(!n) return 1;
    int val = n-> data;
    int val2=0;
    if(n->left) val2+= n->left->data;
    if(n->right) val2+= n->right->data;
    if(val2!=0 && val2!=val) return 0;
    return f(n->left) && f(n->right);
    }
bool isParentSum(Node *root){
    return f(root);
}
```

</details>

### 130. Populate next right pointers of the tree 

<details>

> <img width="542" alt="Screenshot 2024-08-18 at 19 13 25" src="https://github.com/user-attachments/assets/ba3340b2-06f8-4626-9f70-4a84432ffb02">

```cpp
class Solution {
public:
   Node* connect  (Node* r) {
        if(!r) return r;
        queue<Node*> q;
        q.push(r);
        while (!q.empty()) {
            queue<Node*> tmp;
            Node* prev = q.front();
            if(q.front()->left)tmp.push(q.front()->left);
            if(q.front()->right)tmp.push(q.front()->right);
            q.pop();
            long long int cur_size = q.size();
            for (long long int i = 0; i < cur_size; i++) {
               prev->next = q.front();
            if(q.front()->left)tmp.push(q.front()->left);
            if(q.front()->right)tmp.push(q.front()->right);
               prev = q.front();
               q.pop();
            }
            prev->next = NULL;
            if (!tmp.empty()) {               
                while (!tmp.empty()) {
                    q.push(tmp.front());
                    tmp.pop();
                }
            }
        }
        return r;
    }
};
```

</details>

### 131. Search the given key in BST

<details>


```cpp
class Solution {
public:
    TreeNode* f(TreeNode* r, int val){
        if(!r) return NULL;
        if(r->val == val) return r;
        if(r->val<val) return f( r->right , val);
        if(r->val>val) return f( r->left , val);
        return NULL; //dummy
    }
    TreeNode* searchBST(TreeNode* r, int val) {
        return f (r, val);
    }
};
```

</details>

### 132. Construct BST from inorder traversal  

<details>

> The concept is to create a balanced Binary Search Tree (BST) from a sorted array, which represents an inorder traversal of the BST. You can start the process in various ways, such as selecting any index as the root node and then recursively building the left and right subtrees. However, my approach is to always choose the middle index of the current range `[l, r]` as the root. This strategy ensures that the tree remains balanced.

```cpp
class Solution {
public:
    TreeNode* f (vector<int>& n, int l , int h ){
        if(l>h) return NULL;
        int m = (l+h)/2;
        TreeNode* r = new TreeNode (n[m] ) ;
        r->left =  f ( n, l , m-1 );
        r->right = f ( n, m+1 , h );
        return r ;
    }

    TreeNode* sortedArrayToBST(vector<int>& n) {
        return f ( n, 0, n.size() -1 );
    }
};
```

</details>

### 133. Construct BST from preorder traversal

<details>

> To build a Binary Search Tree (BST) from a preorder traversal, you can start by sorting the `pre` vector. By doing this, you effectively get the inorder traversal of the BST. Refer Q.132

> This approach is a bit more complex. Observe that `pre[0]` splits the preorder array into two halves: the left half contains elements smaller than `pre[0]`, and the right half contains elements greater than `pre[0]`. All elements smaller than `pre[0]` after it forms the left subtree JFYI. This observation is helpful, but not directly used in the solution. 

> <img width="583" alt="Screenshot 2024-08-18 at 20 36 26" src="https://github.com/user-attachments/assets/b32cefee-03f7-42d4-9bae-40adf2c1a1d8">


```cpp
class Solution {
public:
    TreeNode* f(vector<int>& pre, long long int ub, int& i){
        if( i == pre.size() || pre[i] > ub  ) return NULL;
        TreeNode* r = new TreeNode( pre[i++] );
        r-> left =  f(pre ,  r->val,i );
        r-> right = f(pre ,  ub ,i);
        return r;
    }
    TreeNode* bstFromPreorder(vector<int>& pre) {
    int i = 0 ;
    return f(pre, 1e18, i );
    }
};
```

</details>

### 134. BT is BST or not 

<details>

> What you need to do is maintain a range `[l, r]` for each node, where the value of the node should fall within this range. Start by initializing the range for the root node to `-1e18` to `1e18`. As you move to the left child, update the right bound of the range to the current node’s value. Conversely, when moving to the right child, update the left bound of the range to the current node’s value. With this approach, you can easily solve the problem.

```cpp
bool f ( TreeNode* rt , long long int l , long long int r ){
    if(!rt) return 1 ;
    if( rt->val <= l || rt->val >= r) return 0;
    return f(rt->left , l , rt->val) && f(rt->right, rt->val , r );
}
class Solution {
public:
    bool isValidBST(TreeNode* root) {
    return f ( root, -1e18, 1e18);    
    }
};
```

</details>

### 135. LCA in BST

<details>

> Refer [Q.119](https://github.com/kuspia/Striver_SDE-shortnotes-/blob/main/README.md#119-lca-in-bt)

> Assuming `p < q`, the idea is to check if both `p` and `q` are smaller than the current node's value; if so, move to the left subtree. If both are greater, move to the right subtree. If the current node’s value falls between `p` and `q`, it is the LCA, as it is the node that creates the paths to `p` and `q`. Additionally, if the current node is either `p` or `q`, directly return that node, since it is an ancestor of the other.

> `p` --- `r` --- `q` 

```cpp
class Solution {
public:
    TreeNode* f(TreeNode* r, TreeNode* p, TreeNode* q){
      if(p->val < r->val  && q->val > r->val) return r; // lca
      if(p == r || q == r ) return r ; // ancestor of the other
      if(p->val < r->val  && q->val < r->val) return f(r->left , p , q);
      else return f(r->right , p , q);
    }
    TreeNode* lowestCommonAncestor(TreeNode* r, TreeNode* p, TreeNode* q) {
        if(p->val < q->val)
        return f(r,p,q);
        else
        return f(r,q,p);
    }
};
```

</details>

### 136. Inorder predecessor and successor of the given key in BST 

<details>

> To find the predecessor of a given key in a Binary Search Tree (BST), you can perform an inorder traversal to obtain a sorted list of all keys. Then, locate the index of the key in this list. The predecessor will be the key at the index immediately before it, while the successor will be at the index immediately after it.

> The time complexity of this approach is `O(H)`, but it can degrade to `O(N)` if the tree is skewed. However, in the case of a balanced BST (like an AVL tree), it's essential to understand this method as it ensures better performance. The idea is to search for the given key and, in each step, decide whether to move left or right. For finding the predecessor, if the current node’s key is less than the target value, update the predecessor. Interestingly, if the current node’s key is equal to the target value, you should still explore the left branch to potentially find a closer immediate predecessor.

```cpp
class Solution
{
    public:
    Node* ans_suc = NULL;
    Node* ans_pre = NULL;
    void f1(Node* r, int val){
        if(!r) return;   
        if(r->key<val) {ans_pre = r;f1( r->right , val);}
        if(r->key>=val) {f1( r->left , val);}
    }
    void f2(Node* r, int val){
        if(!r) return;
        if(r->key <= val) f2( r->right , val);
        if(r->key > val ) { ans_suc = r; f2(r->left, val);} 
    }
    void findPreSuc(Node* r, Node*& pre, Node*& suc, int key)
    {
        f1(r,key);
        pre = ans_pre;
        f2(r,key);
        suc = ans_suc;
    }
};
```

</details>

### 137. Floor in BST

<details>

> Refer Q.136

```cpp
#include <bits/stdc++.h> 
TreeNode<int>* ans_pre = nullptr;
void f1(TreeNode<int>* r, int val) {
    if (!r) return;
    if (r->val == val) {
        ans_pre = r;
        return;
    }
    else if (r->val < val) {
        ans_pre = r;
        f1(r->right, val);
    }
    else f1(r->left, val);
}
int floorInBST(TreeNode<int>* root, int X) {
    ans_pre = nullptr;
    f1(root, X);
    if (ans_pre) {
        return ans_pre->val;
    } else {
        return -1;
    }
}
```

</details>

### 138. Ceil in BST

<details>
	
> Refer Q.136

```cpp
#include <bits/stdc++.h> 
TreeNode<int>* ans_suc = nullptr;
void f1(TreeNode<int>* r, int val) {
    if (!r) return;
    if (r->val == val) {
        ans_suc = r;
        return;
    }
    else if (r->val < val) {
        ans_suc = r;
        f1(r->right, val);
    }
    else f1(r->left, val);
}
int floorInBST(TreeNode<int>* root, int X) {
    ans_suc = nullptr;
    f1(root, X);
    if (ans_suc) {
        return ans_suc->val;
    } else {
        return -1;
    }
}
```

</details>

### 139. Kth smallest element in BST 

<details>

> So what we do is, we do inorder traversal, and simply get kth element 

```cpp
class Solution {
public:
    void f(TreeNode* n, int k, vector<int>& res){
        if(!n) return;
        f(n->left,k,res);
        if(res.size() == k) return;
        res.push_back(n->val);
        f(n->right,k,res);
    }
    int kthSmallest(TreeNode* root, int k) {
        vector<int> res;
        f(root , k, res );
        return res[res.size()-1];
    }
};
```

> Let's optimize the space to O(1), Note here we don't consider recursion stack space, or use Morris traversal to avoid recursion stack space.

```cpp
class Solution {
public:
int res = 0, an;
    void f(TreeNode* n, int k){
        if(!n) return ;
        f(n->left,k);
        if (++res == k) an = n->val;
        f(n->right,k);
    }
    int kthSmallest(TreeNode* root, int k) {
        f(root , k );
        return an;
    }
};
```

> Follow up: If the BST is modified often (i.e., we can do insert and delete operations) then?

> When we have elements greater than the top of `pq` inserted/deleted we don't care, but in case smaller elements come up, remove the top and push the next smaller element, However, if someone smaller than the top is deleted we need to call `f` again to recreate our `pq`.

```cpp
class Solution {
public:
    void f(TreeNode* n, int k, priority_queue<int>& pq){
        if(!n) return ;
        f(n->left,k,pq);
        if(pq.size() < k ) pq.push(n->val);
        if(pq.size() == k) return;
        f(n->right,k,pq);
    }
    int kthSmallest(TreeNode* root, int k) {
        priority_queue<int> pq; // max-heap 
        f(root , k ,pq);
        return pq.top();
    }
};
```

</details>

### 140.  Kth largest element in BST 

<details>

> The idea is to return `N-k+1` smallest node, so we rephrased the problem to which we have already solved. Refer Q.139

```cpp
class Solution {
public:
int res = 0, an;
int N = 0;
    void cnt (Node* r){
        if(!r) return;
        N++;
        cnt(r->left);
        cnt(r->right);
    }
    void f(Node* n, int k){
        if(!n) return ;
        f(n->left,k);
        if (++res == k) an = n->data;
        f(n->right,k);
    }
    int kthLargest(Node *root, int k) {
        cnt(root);
        f(root , N-k+1);
        return an;
    }
};
```

</details>

### 141. Pair with the given sum in BST 

<details>

> M1. To find a pair with a target sum in a Binary Search Tree (BST) using the `O(H)` space, you can leverage the properties of inorder traversal. First, perform an inorder traversal to obtain a sorted array of the BST's values. Then, use a two-pointer approach to find the target sum in this sorted array efficiently.

> Refer to Q.142 then read this, For the BSTIterator approach, maintain two iterators: one for ascending order and one for descending order. This allows you to efficiently find the target sum pair by using two pointers approach appropriately. Special cases include handling nodes with only one child and ensuring that you only call the `next()` function when necessary. Additionally, it’s crucial to check that the iterators are not pointing to the same node to avoid invalid comparisons.

```cpp
class BSTIterator { // L r R
public:
    stack<TreeNode*> s;
    BSTIterator(TreeNode* root) {
        TreeNode* r = root; 
        while(r) {
            s.push(r);
            r = r->left;
        }
    }
    void next() {
        TreeNode* r = s.top();
        s.pop();
        TreeNode* right = r->right;
        while( right  ){
            s.push(right);
            right = right->left;
        }
    }
    bool hasNext() {
      return s.size() ? 1 : 0 ;
    }
};

class BSTIterator1 { // R r L 
public:
    stack<TreeNode*> s;
    BSTIterator1(TreeNode* root) {
        TreeNode* r = root; 
        while(r) {
            s.push(r);
            r = r->right;
        }
    }
    void next() {
        TreeNode* r = s.top();
        s.pop();
        TreeNode* left = r->left;
        while(  left  ){
            s.push(left);
            left = left->right;
        }
    }
    bool hasNext() {
      return s.size() ? 1 : 0 ;
    }
};

class Solution {
public:
    bool findTarget(TreeNode* root, int k) {
        if(!root->right && !root->left) return 0; // just one element is present in it 
        BSTIterator* obj = new BSTIterator(root);
        BSTIterator1* obj1 = new BSTIterator1(root);
        while(obj->hasNext() && obj1->hasNext() && obj->s.top() != obj1->s.top() ){
            if(obj->s.top()->val + obj1->s.top()->val > k) obj1->next();
            else if(obj->s.top()->val + obj1->s.top()->val < k) obj->next();
            else return 1;
        }
        return 0;
    }
};
```

</details>

### 142. BST iterator 

<details>

> The approach is based on simulating the inorder traversal of a BST using a stack. By pushing all left nodes onto the stack during the initialization, we set up the stack to reflect the inorder sequence. When `next()` is called, the top element of the stack represents the next node in the inorder traversal. After retrieving this node, we pop it and then push all left nodes of its right subtree onto the stack. This maintains the inorder property `[LrR]`, ensuring that each call to `next()` returns the nodes in the correct order.

```cpp
class BSTIterator {
public:
    stack<TreeNode*> s;
    BSTIterator(TreeNode* root) {
        TreeNode* r = root; 
        while(r) {
            s.push(r);
            r = r->left;
        }
    }
    int next() {
        TreeNode* r = s.top();
        s.pop();
        TreeNode* right = r->right;
        while(  right  ){
            s.push(right);
            right = right->left;
        }
        return r-> val ;
    }
    
    bool hasNext() {
      return s.size() ? 1 : 0 ;
    }
};

```

</details>

### 143. Size of largest BST in BT 

<details>

> Maintain four variables for each node, as specified in the struct. It’s important to note that if no node is considered, the `max_sum` will always be non-negative, so we initialize it to 0 (the minimum possible value). When processing a node whose left and right subtrees have been fully evaluated, we use the information from both subtrees to determine if the current node forms a valid BST. If it is a valid BST, we update the `max_sum`. Additionally, we need to pass the minimum and maximum values of the current subtree to the previous recursion call.

> Let's try solving the sum of the largest BST in BT first

> <img width="593" alt="Screenshot 2024-08-19 at 12 14 56" src="https://github.com/user-attachments/assets/db6a7c16-3924-4d1c-9a67-e80d56f4f3a6">

> For null node: `(1, 0, ∞, -∞)` and For leaf node: `(1, node->val, node->val, node->val)`

```cpp
class Solution {
public:
    int maxSumBST(TreeNode* root) {
        int maxSum = 0;
        struct Result_dummy = isBST(root, maxSum);
        return maxSum;
    }
private:
    struct Result {
        bool isBST;
        int sum;
        int min_val;
        int max_val;
    };
    Result isBST(TreeNode* node, int& maxSum) {
        if (!node) return {true, 0, INT_MAX, INT_MIN};
        Result left = isBST(node->left, maxSum);
        Result right = isBST(node->right, maxSum);
        int sum = node->val + left.sum + right.sum;
        bool isbst = left.isBST && right.isBST && node->val > left.max_val && node->val < right.min_val;
        if (isbst) maxSum = max(maxSum, sum);
        return {isbst, sum, min(node->val, min (left.min_val , right.min_val ) ), max(node->val , max (left.max_val , right.max_val)  ) };
	// notice every node is not a BST so, we have to return in this fashion 
    }
};
```

> Actual question

```cpp
class Solution {
public:
    int largestBst(Node* root) {
        int maxCnt = 0;
        isBST(root, maxCnt);
        return maxCnt;
    }

private:
    struct Result {
        bool isBST;
        int min_val;
        int max_val;
        int cnt;
    };
    Result isBST(Node* node, int& maxCnt) {
        if (!node) return {true, INT_MAX, INT_MIN, 0};
        Result left = isBST(node->left, maxCnt);
        Result right = isBST(node->right, maxCnt);
	int cnt = left.cnt + right.cnt + 1;
        bool isbst = left.isBST && right.isBST && node->data > left.max_val && node->data < right.min_val;
        if (isbst) maxCnt = max(maxCnt, cnt);
        return {
            isbst, min(node->data, min(left.min_val, right.min_val)), max(node->data, max(left.max_val, right.max_val)), cnt
        };
    }
};

```

</details>

### 144. Serialize And Deserialize a Binary Tree

<details>

> Preorder traversal (easy)

```cpp
class Codec {
public:
    string serialize(TreeNode* root) {
        stringstream ss;
        serializeHelper(root, ss);
        return ss.str();
    }
    TreeNode* deserialize(string data) {
        stringstream ss(data);
        return deserializeHelper(ss);
    }

private:
    void serializeHelper(TreeNode* root, stringstream& ss) {
        if (root == nullptr) {
            ss << "# "; 
            return;
        }
        ss << root->val << " ";
        serializeHelper(root->left, ss);
        serializeHelper(root->right, ss);
    }

    TreeNode* deserializeHelper(stringstream& ss) {
        string val;
        ss >> val;
        if (val == "#") {
            return nullptr;
        }
        TreeNode* node = new TreeNode(stoi(val));
        node->left = deserializeHelper(ss);
        node->right = deserializeHelper(ss);
        return node;
    }
};

```

</details>

### 145. BT to DLL

<details>


```cpp
```

</details>

### 146. The median in the stream of running integers

<details>


```cpp
```

</details>

### 147. Kth largest element in a stream 

<details>


```cpp
class KthLargest {
public:
    KthLargest(int k, vector<int>& nums) { // first we will have nums array over which new integers will come up
        this->k = k;
        for (int num : nums) int dummy = add(num);  
    }
    int add(int val) {
        if (pq.size() < k) pq.push(val);
        else if (val > pq.top()) { 
            pq.pop();
            pq.push(val);
        }
        return pq.top();
    }
    
private:
    int k;
    priority_queue<int, vector<int>, greater<int>> pq;
};
```

</details>

### 148. Distinct numbers in windows 

<details>


```cpp
```

</details>

### 149. Kth largest element in the unsorted array

<details>

> Kth largest => min-heap DS (memory tip)

```cpp
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue <int , vector<int> , greater<int> > pq;
        for(auto i : nums ){
            if(pq.size() < k) pq.push(i);
            else{
                if( pq.top() < i ){
                    pq.pop();
                    pq.push(i);

                }
            } 
        }
        return pq.top();
    }
};
```

</details>

### 150. Flood fill Algorithm 

<details>

> BFS easy

```cpp
class Solution {
public:
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
        int rows = image.size();
        int cols = image[0].size();
        int originalColor = image[sr][sc];   
        if (originalColor == color) return image;
        queue<pair<int, int>> q;
        q.push({sr, sc});
        vector<int> dx = {-1, 1, 0, 0};
        vector<int> dy = {0, 0, -1, 1};
        while (!q.empty()) {
            pair<int, int> current = q.front();
            q.pop();
            int x = current.first;
            int y = current.second;
            image[x][y] = color;
            for (int i = 0; i < 4; i++) {
                int newX = x + dx[i];
                int newY = y + dy[i];
                if (newX >= 0 && newX < rows && newY >= 0 && newY < cols && image[newX][newY] == originalColor) q.push({newX, newY}); 
            }
        }
        return image;
    }
};
```

> DFS

```cpp
class Solution {
public:
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        int originalColor = image[sr][sc];
        if (originalColor != newColor) dfs(image, sr, sc, originalColor, newColor);
        return image;
    }

private:
    vector<int> dx = {-1, 1, 0, 0};
    vector<int> dy = {0, 0, -1, 1};
    void dfs(vector<vector<int>>& image, int x, int y, int originalColor, int newColor) {
        image[x][y] = newColor; // Change the color
        for (int i = 0; i < 4; i++) {
            int newX = x + dx[i];
            int newY = y + dy[i];
            if (newX >= 0 && newX < image.size() && newY >= 0 && newY < image[0].size() && image[newX][newY] == originalColor) {
                dfs(image, newX, newY, originalColor, newColor);
            }
        }
    }
};
```

</details>

### 151. Clone a graph (hard)

<details>


```cpp
```

</details>

### 152. DFS

<details>


```cpp
```

</details>

### 153. BFS

<details>


```cpp
```

</details>

### 154. Detect Cycle in an Undirected Graph (using BFS)

<details>


```cpp
```

</details>








