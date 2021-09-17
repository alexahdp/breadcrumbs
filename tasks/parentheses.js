/**
 * @param {number} n
 * @return {string[]}
 */
var generateParenthesis = function(n) {
    const res = [];
    backtrack([], 0, 0);
    
    function backtrack(str, left, right) {
        if (str.length === 2*n) {
            res.push(str.join(''));
            return
        }
        
        if (left < n) {
            str.push('(');
            backtrack(str, left+1, right);
            str.pop();
        }
        if (right < left) {
            str.push(')');
            backtrack(str, left, right+1);
            str.pop();
        }
    }
    return res;
}
