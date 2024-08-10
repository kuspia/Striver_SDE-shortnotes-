# Striver_SDE sheet solution

1. Set matrix zero

> You can create a lookup row and col for each cell, the idea is to suppose we are allowed to take extra space and then create row_lookup[c] and col_lookup[r], size is shown in the [], now just iterate and wherever you find 1 mark corresponding position in both lookup tables, finally reiterate the whole matrix and see if you have marked in any of the lookup tables from the both if yes set that cell to 1.

> However we try to save the space by utilizing the 0th r/c of my matrix as a lookup table but to explicitly deal with the mat[0][0] (as it overlaps in both) we have denoted it with the help of the row/col variable.

```
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
2. Pascal Triangle

> n space problem (easy)

```
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
3. Next Permutation

> The idea is based upon the fact that always remember that when we want to create the next permutation we try to choose the next value as close as possible to the given value, (basically a next greater permutation), ex: 1 3 4 2

> So when you are at some index, and you want to increase it, then ask yourself the numbers lying after it forms the largest number? if yes you can increment the index to the closest possible value that is lying on the right-hand side of it and choose the next possible value otherwise move on from that index towards the right.

> Ex: 1 3 5 4 2: you should move on from 0th index -> you should increment the 1st index since 542 forms the largest possible number among {2, 4, 5} -> 1 4: 5 3 2 ->  L: R (reverse R to get the final number) -> 1 4 2 3 5

> see the code its little tricky bro !!

```
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

4. Kadane Algo: Find the subarray with the largest sum, and return its sum.

> The only thing that you should understand here is we break the subarray as soon as the overall sum becomes **-ve**, the only reason for doing so is that taking some -ve sum won't help us further because if you add it you will just decrease the overall maximizing sum that we need. When every element is negative has been handled explicitly.

```
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

5. Sort array having 0,1,2 (three elements only)

> Just remember we need 3 pointers, `l/m/h` and yes `m` is like an iterator throughout the array while `l` and `h` are useful to mark the boundary for `0` and `1` respectively.

```
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

6. Stock Buy and Sell [SP - CP]: You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. 

> Make a Graph and observe that u have to choose largest slant line inclined towards right, (left is not possible as we need to choose to buy and then sell), Please note two points can be connected from any one to another to observe the required line which we are looking for,

> While traversing the array if SP - CP turns out to be negative better ignore it (since there is option not to sell the stock If you cannot achieve any profit, return 0) and while you traverse since we want to expand the length of our line, so its better if we go deeper at SP end, this is reason why we have minsofar variable.

```
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

7. Rotate the matrix 90 degrees clockwise

> swap(a[i], a[j]) and then reverse every row.

> observe we did this in loop `j < i` reason is we want to traverse lower triangular matrix only.

```
void rotate(vector < vector < int >> & matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) swap(matrix[i][j], matrix[j][i]);
    for (int i = 0; i < n; i++) reverse(matrix[i].begin(), matrix[i].end());
}
```

8. Merge overlapping subintervals

> Sort `iv` based on the first element and then the second element, in your hand, pick the first interval and say it is `cur` now with further intervals [1 to iv. size()-1]. Check whether there is overlap, if yes try expanding the cur interval to max boundary points else push it to the answer and choose a new interval as your `cur`.

```
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
9. Merge 2 sorted array (no extra space)

> To do it in `O(n+m)` you can take the extra space and do it, however, that approach is similar to what we did in merge sort, but to do it in 1 space we need `O(nlgn)` time, now we assume that `a` will keep the smallest element at [0] while `b` keeps the largest element at last in it, so please remember the initialization which has utmost importance = Place the pointer at the last of `a` and at the beginning of the array `b`. A way to remember is take two array so when you join them, pointers are supposed to be placed at the joints.

```
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

10. Find duplicate in array of N+1 integers

> Refer Q.11 M3

```
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

11. Repeat and missing number

> M1: Let the array size be n, so as per the expectations 1 to n numbers were supposed to be present in it. However, one guy is missing while the other guy is being repeated. We use BF to find the missing/repeating number by having doubt over every single number from 1 to n  and then checking it with the given array. Clearly, it takes `O(2*n^2)`.

> M2: Use the hash map, say 1 to n is my index of an array whenever we find a number we do ++ over there, clearly when you reiterate the hash array and if you find somewhere 0 b/w 1 to n that means that index is my missing number, a corollary to that is if you find somewhere as 2 stored that means that index is being repeated.

> M3: Iterate the array and just imagine that array is 1 indexed, now your job is that for every index `i`, go to a[i]th index and mark the value stored with a `-ve` sign, however, if it is already marked that means a[i] is my repeating number, in that case don't mark it, and kindly continue the process, at the end you will see that one of the ith index is still having a `+ve` sign that index is actually the missing one.  

> M4: Maths, say the sum of the array is s1 while the sum of squares of elements is s2, say y is your repeating number while x is the missing, we derive a formula as `x-y = n(n+1)/2 -s1` and `x^2 - y^2 = n(n+1)(2n+1)/6 - s2`, clearly you have x-y and x+y so congratulations!!

> M5: This is kind of unobvious, an XOR trick, so take the xor of all array elements and then keep on taking the xor till 1 to n now at the end you will be left with a xor value which actually represents x^y, now we need to segregate the x and y from x^y which is very tricky, please see the code, and understand yourself.

```
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

11.1. Extension problem: Imagine if you had elements from 0 to n-1 and an array of size n, then let's say elements, appear more than once like n=10: [0 1 0 0 1 1 5 9 9 0], how do we find all duplicates and missing exactly once in `n` time and `1` space?  Here what we do is we go to arr[i] index, and increase that index by n every time, the rest code is self-explanatory.

```
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

12. Inversion of array [explanation1](https://www.youtube.com/watch?v=AseUmwVNaoY&t=364s) [mergesort-vid-animation](https://www.youtube.com/watch?v=5Z9dn2WTg9o)

> You should note that we recur for the left and right parts so that also gives me an inversion count we need to add both and when you merge the two halves at any step it also generates the count so we need to add up the count for three cases, now when we go to merge and we say while merging if a guy from left half becomes greater than any guy on the right then everyone following the guy on the left side including it will be greater than the right half guy, so that's what is used to count the inversions. `cnt += (mid - left + 1); //Trick`

```
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

13. Search in 2D matrix

```
```

15. Pow(x,n)

```
```

17. Majority (>n/2)

```
```

19. Majority (>n/3)

```
```

21. Grid unique paths

```
```

23. Reverse pairs

> Refer Q.12

```
```





























   
