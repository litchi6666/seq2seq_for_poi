def isPalindrome(x):
    """
    :type x: int
    :rtype: bool
    """
    if x < 0 or (x % 10 == 0 and x > 0):
        return False

    re = 0
    while x > re:
        re = re * 10 + x % 10
        x = x//10

    return x==re or x==re//10


def letterCombinations(digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    dict_ = {'1':'','2':"abc",'3':'def','4':'ghi','5':'jkl',
            '6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    all_result = list(dict_[digits[0]])
    for i in range(1,len(digits)):
        temp = []
        for char_a in all_result:
            for char_b in dict_[digits[i]]:
                temp.append(char_a+char_b)
        all_result = temp
    return all_result


def isValid(s):
    dict = {')':'(',']':'[','}':'{'}
    slist = list(s)
    stack = []
    for i in range(len(slist)):
        if len(stack)>0 and dict.get(slist[i],0) == stack[-1]:
            stack.pop()
        else:
            stack.append(slist[i])
    if len(stack)>0:
        return False
    else:
        return True


def fourSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    nums = sorted(nums)
    length = len(nums)
    result = []

    if length < 3:
        return result

    last_i = 0
    for i in range(length-3):
        if not i == 0 and nums[last_i] == nums[i]:
            continue
        # 当前i和上一个i相同 跳过
        last_i = i

        last_j = i+1
        for j in range(i+1, length-2):
            # 当前j和上一个j相同 跳过
            if not j == i+1 and nums[last_j] == nums[j]:
                continue
            last_j = j

            two_sum = nums[i]+nums[j]

            l = j + 1
            r = length - 1

            while l < r:
                four_sum = two_sum + nums[l] + nums[r]
                if four_sum == target:
                    temp = [nums[i], nums[j], nums[l], nums[r]]
                    result.append(temp)
                    while l<r and nums[l] == nums[l+1]:
                        l += 1
                    while l<r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
                if four_sum > target:
                    r -= 1
                if four_sum < target:
                    l += 1

    return result

    # # print(fourSum([1, 0, -1, 0, -2, 2],target = 0))
    # print(fourSum([-1,0,-5,-2,-2,-4,0,1,-2],-9))
    # print(fourSum([0,0,0,0],0))

def threeSumClosest(nums, target):
    """
    :type nums: List[int] nums = [-1，2，1，-4], 和 target = 1.
    :type target: int
    :rtype: int
    """
    nums = sorted(nums)
    length = len(nums)

    def sum(a,b,c):
        return nums[a]+nums[b]+nums[c]

    import sys
    min_dist = sys.maxsize

    result = 0
    for i in range(length-2):
        l = i + 1
        r = length - 1

        while l < r:
            sum_3 = sum(i,l,r)
            if min_dist > abs(sum_3 - target):
                min_dist = abs(sum_3 - target)
                result = sum_3
            if sum_3 < target:
                l += 1
            elif sum_3 > target:
                r -= 1
            else:
                return sum_3

    return result

def twoSum(nums, target):
    list = []
    for i in range(len(nums)):
        to_find = target - nums[i]
        temp = nums[i]
        nums[i] = to_find + 1

        if to_find in nums:
            j = nums.index(to_find)
            list.append(i)
            list.append(j)
        nums[i] = temp

        return list
    return list

def strStr(haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    if len(needle) < 1:
        return 0
    for i in range(len(haystack)-len(needle)+1):
        if haystack[i] == needle[0]:
            j = 0
            while haystack[i + j] == needle[j]:
                if j == len(needle)-1:
                    return i
                j += 1
    return -1


print(strStr('','aa'))
