client='xweichu@c220g2-011113.wisc.cloudlab.us'
ssh -p 22 $client << 'EOF'
    sudo apt-get --assume-yes update;
    sudo apt-get --assume-yes install docker.io;
    sudo apt-get --assume-yes install vim;
    sudo apt-get --assume-yes install ceph-fuse;
    sudo apt-get --assume-yes install fio
    sudo apt-get --assume-yes install gnuplot
    sudo apt-get --assume-yes install ceph-common
    sudo apt-get --assume-yes install ceph-fs-common
    sudo apt-get --assume-yes install attr;
    sudo mkdir -p /etc/ceph/
    sudo mkdir -p /var/lib/ceph/bootstrap-osd/
    sudo mkdir /mnt/cephfs/
    sudo chmod -R 777 /etc/ceph/
    sudo chmod -R 777 /var/lib/ceph/bootstrap-osd/
    sudo chmod -R 777 /mnt/cephfs/
EOF

scp ./ceph_keys/* $client:/etc/ceph
scp ./ceph_keys/* $client:/var/lib/ceph/bootstrap-osd
scp ~/Google\ Drive/Personal/Key/* $client:/users/xweichu/.ssh

ssh -p 22 $client << 'EOF'
    sudo ceph-fuse -k /etc/ceph/ceph.client.admin.keyring -c /etc/ceph/ceph.conf /mnt/cephfs
    eval "$(ssh-agent -s)"
    ssh-add -k ~/.ssh/id_rsa
    sudo chmod 777 /mnt/cephfs/
    # scp cross@pulpo-dtn.ucsc.edu:/mnt/pulpos/cross/wdmerger_for_ucsc/* /mnt/cephfs
    # scp cross@pulpo-dtn.ucsc.edu:/mnt/pulpos/cross/wdmerger_for_ucsc_2/* /mnt/cephfs
    # sudo mkdir /mnt/cephfs/all_tar
    # sudo chmod 777 /mnt/cephfs/all_tar/
    # ls /mnt/cephfs/*.tar | xargs -i tar xf {} -C /mnt/cephfs/all_tar/

EOF