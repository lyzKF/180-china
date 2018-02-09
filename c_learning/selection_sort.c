/**********************************************************
 * Author        : lyz
 * Email         : lyz038015@163.com
 * Create time   : 2018-02-03 00:44
 * Last modified : 2018-02-03 00:44
 * Filename      : selection_sort.c
 * Description   : 选择排序
 * *******************************************************/
#include <stdio.h>
void selection_sort(int *a, int n);

int main(void)
{
    int test[5] = {1,5,3,9,0};
    selection_sort(test, 5);
    int k;
    for (k=0;k<5;k++)
    {
	printf("%d\n",test[k]);
    }
    return 0;
}

void selection_sort(int *a, int n) //a为数组指针，n为数组长度
{
    int i,j,temp;
    int min;
    for (i=0; i<n-1; i++)
    {
	min = i; //元素索引号
	for (j=i+1; j<n; j++)
	{
	    if (a[j] < a[min])
	    {
		min = j;
	    }
	}
	temp = a[i];
	a[i] = a[min];
	a[min] = temp;
    }

}
