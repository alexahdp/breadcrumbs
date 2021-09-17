/**
You are given an array of distinct integers arr and an array of integer arrays pieces, where the integers in pieces are distinct. Your goal is to form arr by concatenating the arrays in pieces in any order. However, you are not allowed to reorder the integers in each array pieces[i].

Return true if it is possible to form the array arr from pieces. Otherwise, return false.


Example 1:

Input: arr = [85], pieces = [[85]]
Output: true
Example 2:

Input: arr = [15,88], pieces = [[88],[15]]
Output: true
Explanation: Concatenate [15] then [88]
Example 3:

Input: arr = [49,18,16], pieces = [[16,18,49]]
Output: false
Explanation: Even though the numbers match, we cannot reorder pieces[0].
Example 4:

Input: arr = [91,4,64,78], pieces = [[78],[4,64],[91]]
Output: true
Explanation: Concatenate [91] then [4,64] then [78]
Example 5:

Input: arr = [1,3,5,7], pieces = [[2,4,6,8]]
Output: false


Constraints:

1 <= pieces.length <= arr.length <= 100
sum(pieces[i].length) == arr.length
1 <= pieces[i].length <= arr.length
1 <= arr[i], pieces[i][j] <= 100
The integers in arr are distinct.
The integers in pieces are distinct (i.e., If we flatten pieces in a 1D array, all the integers in this array are distinct).
*/


/**
 * @param {number[]} arr
 * @param {number[][]} pieces
 * @return {boolean}
 */
var canFormArray = function(arr, pieces) {
    const arr2map = (arr) => ({
        [arr[0]]: arr.length > 2 ? arr2map(arr.slice(1)) : arr[1],
    });
    
    // собираем из pieces объект
    // идем по arr и проверяем поштучно наличие ключей в объекте
    const piecesMap = pieces.reduce((o, a) => {
        if (a.length === 1) {
            o[a[0]] = true;
        }
        else if (a.length === 2) {
            o[a[0]] = a[1];
        }
        else {
            o[a[0]] = arr2map(a.slice(1));
        }
        return o;
    }, {});
    
    const seek = (arr, j, piecesMap) => {
        let cur = piecesMap;
        while (true) {
            if (typeof cur === 'number') {
                if (cur !== arr[j]) return j;
                return ++j;
            }
            else if (cur === true) {
                return j;
            }
            else {
                if (!cur[arr[j]]) return j;
                cur = cur[arr[j]];
                j++;
            }
            if (j >= arr.length) return j;
        }
    };
    
    let i = 0;
    while(i < arr.length) {
        const n = seek(arr, i, piecesMap);
        if (i === n) return false;
        i = n;
    }
    return true;
};
