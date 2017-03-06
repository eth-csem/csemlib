/* add_crust for all parameters */


void add_crust(long long int npoints, double crust_dep[], double crust_vs[], double topo[], double vsv[], \
                double vsh[], double vpv[], double vph[], double rho[], double rad[])
{
    // Locally allocated
    double r_earth = 6371.0;
    double r_ani = 6191.0;
    double s_ani = 0.0011;
    double taper_percentage = 0.25;
    double vsv_crust;
    double vsh_crust;
    double taper_hwidth;
    double taper_width;
    double frac_crust;
    double frac_mantle;
    double dist_from_mantle;

    long long int i;
    for (i=0; i<npoints; i++) {
        taper_hwidth = crust_dep[i] * taper_percentage;

        // If above taper, overwrite with crust.
        if (rad[i] > (r_earth - crust_dep[i] + taper_hwidth)) {
            vsv[i] = crust_vs[i] - 0.5 * s_ani * (rad[i] - r_ani);
            vsh[i] = crust_vs[i] + 0.5 * s_ani * (rad[i] - r_ani);

            //Scaling to P velocities and density for continental crust.
            if (topo[i] >= 0.0) {
                vpv[i] = 1.5399 * crust_vs[i] + 0.840;
                vph[i] = 1.5399 * crust_vs[i] + 0.840;
                rho[i] = 0.2277 * crust_vs[i] + 2.016;
            }

            //Scaling to P velocities and density for oceanic crust.
            if (topo[i] < 0.0) {
                vpv[i] = 1.5865 * crust_vs[i] + 0.844;
                vph[i] = 1.5865 * crust_vs[i] + 0.844;
                rho[i] = 0.2547 * crust_vs[i] + 1.979;
            }
        }
        // If below taper region in mantle, do nothing and continue.
        else if (rad[i] < (r_earth - crust_dep[i] - taper_hwidth)) {
            continue;
        }
        // In taper region, taper linearly to mantle properties.
        else {
            dist_from_mantle = rad[i] - (r_earth - crust_dep[i] - taper_hwidth);
            taper_width = 2.0 * taper_hwidth;
            frac_crust = dist_from_mantle / taper_width;
            frac_mantle = 1.0 - frac_crust;

            // Ascribe crustal vsh and vsv based on the averaged vs by Meier et al. (2007).
            vsv_crust = crust_vs[i] - 0.5 * s_ani * (rad[i] - r_ani);
            vsv[i] = (vsv_crust * frac_crust) + (vsv[i] * frac_mantle);
            vsh_crust = crust_vs[i] + 0.5 * s_ani * (rad[i] - r_ani);
            vsh[i] = (vsh_crust * frac_crust) + (vsh[i] * frac_mantle);

            // Scaling to P velocities and density for continental crust.
            if (topo[i] >= 0.0 ) {
                vpv[i] = (1.5399 * crust_vs[i] + 0.840) * frac_crust + (vpv[i] * frac_mantle);
                vph[i] = (1.5399 * crust_vs[i] + 0.840) * frac_crust + (vph[i] * frac_mantle);
                rho[i] = (0.2277 * crust_vs[i] + 2.016) * frac_crust + (rho[i] * frac_mantle);
            }

            // Scaling to P velocity and density for oceanic crust.
            if (topo[i] < 0.0 ) {
                vpv[i] = (1.5865 * crust_vs[i] + 0.844) * frac_crust + (vpv[i] * frac_mantle);
                vph[i] = (1.5865 * crust_vs[i] + 0.844) * frac_crust + (vph[i] * frac_mantle);
                rho[i] = (0.2547 * crust_vs[i] + 1.979) * frac_crust + (rho[i] * frac_mantle);
            }
        }
    }
}

