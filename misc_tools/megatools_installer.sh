# The dataset is hosted on mega.nz servers. Megatools would be required to download data from these servers to Unix boxes.
# To gain access to the dataset, you'll have to register with endoscopic-vision website and join the roboticinstrumentsegmentation group. You can register using your Gmail account, and join the group by clicking on 'Join' (the rightmost option on the bar below the logo)
sudo apt-get update
sudo apt-get install libtool libglib2.0-dev gobject-introspection libgmp3-dev nettle-dev asciidoc glib-networking libcurl4-gnutls-dev libssl-dev
sudo dpkg --configure -a

wget http://megatools.megous.com/builds/megatools-1.9.98.tar.gz
zcat megatools-1.9.98.tar.gz > megatools-1.9.98.tar
tar -xf megatools-1.9.98.tar
cd megatools-1.9.98/
./configure
make
sudo make install
# use the URL's mentioned in the downloads page to download data to Unix using the command: megadl <URL>
# Copy data to project dir with structure as shown in https://github.com/ternaus/robot-surgery-segmentation