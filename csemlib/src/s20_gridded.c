/* Hello World program */

#include<stdio.h>
//#include <stdlib.h>     /* abs */



double lagrange(double x, double x0, double x1, double x2)
{
    return ((x-x1)/(x0-x1))*((x-x2)/(x0-x2));
}

int find_idx_closest(long long int array_length, double value, double array[])
{
    // Initialize Variables
    double EPS = 1E-15;
    long long int idx = 0;
    double difference;
    difference = fabs(value - array[idx]);
    long long int i;

    // Search for index of nearest value
    for (i=0; i<array_length; i++) {
        if (fabs(value-array[i]) < difference) {
            difference = fabs(value-array[i]);
            idx = i;
            }
        if (difference < EPS) {
            break;
        }
    }
    return idx;
}

void s20eval_grid(long long int len_c, long long int len_l, long long int len_r, long long int n, double c[], double l[], double r[],  \
                    double colat[], double lon[], double rad[], double dv_out[], double dv[][len_c][len_l])
{
    // Declare internal variables
    long long int ic, il, ir, irm, irp, ilm, ilp;
    double Ll0mp, Llm0p, Llpm0, Lc0mp, Lcm0p, Lcpm0, colat_i, lon_i, rad_i;

    long long int i;
    for (i=0; i<n; i++) {
        colat_i = colat[i];
        lon_i = lon[i];
        rad_i = rad[i];

        // Find Nearest indices
        ic = find_idx_closest(len_c, colat_i, c);
        il = find_idx_closest(len_l, lon_i, l);
        ir = find_idx_closest(len_r, rad_i, r);

        // Check this out with andreas, I think It was wrong, so I changed it...
        if (r[ir] > rad_i) {
            irm = ir;
            irp = ir + 1;
        }
        else {
            irm = ir - 1;
            irp = ir;
        }

        // Weights for linear depth interpolation.
        double m = (rad_i-r[irp])/(r[irm]-r[irp]);
        double p = (rad_i-r[irm])/(r[irp]-r[irm]);

        if ((ic > 0) && (ic < (len_c - 1))){
            if (il == 0){
                ilm = len_l - 1;
            }
            else {
                ilm = il - 1;
            }
            if (il == (len_l - 1)){
                ilp = 0;
            }
            else {
                ilp = il + 1;
            }
            // Precompute terms for lagrange interpolation
            if ((i == 0 ) || (lon_i != lon[i-1])){
                Ll0mp=lagrange(lon_i,l[il],l[ilm],l[ilp]);
                Llm0p=lagrange(lon_i,l[ilm],l[il],l[ilp]);
                Llpm0=lagrange(lon_i,l[ilp],l[ilm],l[il]);
            }
            if ((i == 0) || (colat_i != colat[i-1])){
                Lc0mp=lagrange(colat_i,c[ic],c[ic-1],c[ic+1]);
                Lcm0p=lagrange(colat_i,c[ic-1],c[ic],c[ic+1]);
                Lcpm0=lagrange(colat_i,c[ic+1],c[ic-1],c[ic]);
            }
            // Lagrange interpolation.
            dv_out[i] = (p * dv[irp][ic][il] + m * dv[irm][ic][il]) * Lc0mp * Ll0mp + \
                        (p * dv[irp][ic-1][il] + m * dv[irm][ic-1][il]) * Lcm0p * Ll0mp + \
                        (p * dv[irp][ic+1][il] + m * dv[irm][ic+1][il]) * Lcpm0 * Ll0mp + \
                        (p * dv[irp][ic][ilm] + m * dv[irm][ic][ilm]) * Lc0mp * Llm0p + \
                        (p * dv[irp][ic-1][ilm] + m * dv[irm][ic-1][ilm]) * Lcm0p * Llm0p + \
                        (p * dv[irp][ic+1][ilm] + m * dv[irp][ic+1][ilm]) * Lcpm0 * Llm0p + \
                        (p * dv[irp][ic][ilp] + m * dv[irm][ic][ilp]) * Lc0mp * Llpm0 + \
                        (p * dv[irp][ic-1][ilp] + m * dv[irm][ic-1][ilp]) * Lcm0p * Llpm0 + \
                        (p * dv[irp][ic+1][ilp] + m * dv[irm][ic+1][ilp]) * Lcpm0 * Llpm0;

        }
        else {
            dv_out[i] = (p *dv[irp][ic][il] + m * dv[irm][ic][il]);
        }
    }
}

/*
int main(){

    // INPUTS
    double c[] = {2.5, 5.0, 7.5};
    double l[] = {20, 25.0, 30.0};
    double r[] = {6200.0, 6210.0, 6220.0};
    // toy dv
    double dv[3][3][3] = {
            {{2, 1,2}, {3, 4, 5}, {2, 1,2}},
            {{5, 6, 7}, {9, 10, 11}, {2, 1,2}},
            {{1, 1,2}, {3, 4, 5}, {2, 1,2}},
            };
    //double dv[len_c][len_r][len_l];
    long long int len_c = sizeof(c) / sizeof(c[0]);
    long long int len_l = sizeof(l) / sizeof(l[0]);
    long long int len_r = sizeof(r) / sizeof(r[0]);
    double colat[] = {5.0, 3.0};
    double lon[] = {21.0, 24.0};
    double rad[] = {6205.0, 6208.0};

    long long int n = sizeof(colat) / sizeof(colat[0]); // get this from python as well
    double dv_out[n];

    s20eval_grid(len_c, len_l, len_r, n, c, l, r, colat, lon, rad, dv_out, dv);
    long long int ii = 0;
        for (ii=0; ii< n; ii++){
            printf("%f \n", dv_out[ii]);
        }

     //this one as well
    // End of inputs

}

*/
