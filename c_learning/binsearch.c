/**********************************************************
 * Author        : lyz
 * Email         : lyz038015@163.com
 * Create time   : 2018-02-03 19:07
 * Last modified : 2018-02-03 19:32
 * Filename      : binsearch.c
 * Description   : 折半查找
 * *******************************************************/

#include <stdio.h>
int bin_search(int *a, int n, int x);

int main(void)
{
    int temp[] = {1,2,3,4,8,10};
    int result = 0;
    result = bin_search(temp, 6, 3);
    if (result == -1)
    {
	printf("None\n");
    }
    else
    {
	printf("目标对象位置:%d\n",result);
    }
    return 0;
}

int bin_search(int *a, int n, int x)
{
    /* a为数组指针，其已经有顺序
     * n为数组的长度
     * x为查找对象
    */
    int low = 0;
    int high = n-1;
    int mid = 0;
    while (low <= high)
    {
	mid  = (low + high) / 2;
	if (x < a[mid])
	{
	    high = mid;
	}
	else if (x > a[mid])
	{
	    low = mid;
	}
	else if (x == a[mid])
	{
	    return mid;
	}
    }
    return -1;
}
