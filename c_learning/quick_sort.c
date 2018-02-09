/**********************************************************
 * Author        : lyz
 * Email         : lyz038015@163.com
 * Create time   : 2018-02-09 21:51
 * Last modified : 2018-02-09 21:51
 * Filename      : quick_sort.c
 * Description   : 
 * 快速排序算法是一种基于交换的高效排序算法，其采用了分治法的思想。
 * 1. 从数列中取出一个数作为基准数(pivot)。
 * 2. 将数组进行划分(partition)，将比基准数大的元素移至pivot右边；小于等于基      准数的元素移至pivot左边。
 * 3. 对左右区间重复第二步的划分，直至每个子区间只有一个元素。
 * *******************************************************/
# include <stdio.h>
void Quick_sort(int *arr, int left, int right);
int partition(int *arr, int left, int right);

int main(void)
{
    int a[] = {1,3,5,2,7,6};
    Quick_sort(a,0,6);
    int index;
    for(index = 0; index<6; index++)
    {
	printf("%d\n",a[index]);
    }
    return 0;
}

int partition(int *arr, int left, int right)
{
    int i = left, j = right;
    int pivot = arr[left];
    while(i < j)
    {
	// 从右向左寻找，大于pivot的数
	while(i<j && arr[j] > pivot)
	{
	    j--;
	}
	if(i < j)
	{
	    arr[i] = arr[j];
	    i += 1;
	}
	// 从左向右寻找，小于pivot的数
	while(i<j && arr[i] < pivot)
	{
	    i++;
	}
	if (i<j)
	{
	    arr[j] = arr[i];
	    j -= 1;
	}
    }
    //
    arr[i] = pivot;
    return i;
}

void Quick_sort(int *arr, int left, int right)
{
    if (left > right)
    {
	return ;
    }
    int j = partition(arr,left,right);
    Quick_sort(arr, left, j-1);
    Quick_sort(arr, j + 1, right);
}
