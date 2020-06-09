/**
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
Example:
  Given nums = [2, 7, 11, 15], target = 9,
  Because nums[0] + nums[1] = 2 + 7 = 9,
  return [0, 1].
*/

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
var twoSum1 = function(nums, target) {
    const dict = nums.reduce((o, n, i) => Object.assign(o, {[n]: i}), {});
    for (let i = 0; i < nums.length; i++) {
        const j = dict[target - nums[i]];
        if (j && i !== j) return [i, j];
    }
};

// This is better solution
var twoSum = function(nums, target) {
    const dict = {};
    for (let i = 0; i < nums.length; i++) {
        if (dict[nums[i]] >= 0) return [dict[nums[i]], i];
        dict[target - nums[i]] = i;
    }
};
