subroutine sph2v(nPoints, lat, lon, dep, dv_d, mfl, wasread)
      implicit none
      INTEGER nPoints
      integer, parameter       :: rk=SELECTED_REAL_KIND(15, 307)
      integer :: nNodes, i
      real(kind=rk), dimension(nPoints)  :: lat
      real(kind=rk), dimension(nPoints)  :: lon
      real(kind=rk), dimension(nPoints)  :: dep
      real(kind=rk), dimension(nPoints)  :: dv_d
      real, dimension(nPoints)  :: dv
      real(kind=rk), parameter :: R_EARTH=6371
      logical :: wasread
      character*80 :: mfl

!f2py intent(in) :: nPoints, lat, lon, dep, dv_d, wasread, mfl
!f2py intent(out) :: dv_d

      do i = 1, nPoints
        call sph2v_sub(real(lat(i)), real(lon(i)), &
             real(dep(i)), dv(i), mfl, wasread)
        wasread = .true.
      end do

!     convert to double
      dv_d    = real(dv, rk) 
end

