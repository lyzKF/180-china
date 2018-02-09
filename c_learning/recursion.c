/**********************************************************
 * Author        : lyz
 * Email         : lyz038015@163.com
 * Create time   : 2018-02-03 19:48
 * Last modified : 2018-02-03 19:48
 * Filename      : factorial.c
 * Description   : factorial by recursion
 * *******************************************************/
# include <stdio.h>
long factorial(int n);

int main(void)
{
    int n = 10;
    long result = 0;
    result = factorial(n);
    printf("%d! = %ld\n",n,result);
    return 0;
}

long factorial(int n)
{
    if (n ==0)
    {
	return 1;
    }
    else
    {
	return n * factorial(n-1);
    }
}
