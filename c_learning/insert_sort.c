/**********************************************************
 * Author        : lyz
 * Email         : lyz038015@163.com
 * Create time   : 2018-02-03 23:56
 * Last modified : 2018-02-04 00:06
 * Filename      : insert_sort.c
 * Description   : 
 * *******************************************************/
#include <stdio.h>
void insert_sort(int *arr,int n);

int main(void)
{
    int i;
    //
    int arr[5] = {1,3,6,4,2};
    insert_sort(arr, 5);
    printf("#");
    for(i=0;i<5;i++)
    {
	printf("%4d\n",arr[i]);
    }
    return 0;
}

void insert_sort(int *arr,int n)
{
    int i,j;
    int temp = 0;
    for(i = 0;i < n; i++)
    {
	// 取出一个元素
	temp = arr[i];
	// 对取出的元素与其前面的元素做对比
	for(j = i; j>0 && temp < arr[j-1]; j--)
	{
	    // 如果前面的元素较大，则后移一位
	    if(temp < arr[j-1])
	    {
		arr[j] = arr[j-1];
	    }
	}
	// 取出的元素填充到前面的元素空位
	arr[j] = temp;
    }
}
