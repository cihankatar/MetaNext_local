
bind_addr = "0.0.0.0"
datacenter = "dc1"

acl {
  enabled = true
}

log_level  = "DEBUG"
data_dir   = "/tmp/local_server"
name       = "local_server"

server {
  enabled          = true
  bootstrap_expect = 1
  num_schedulers   = 1
}

client {
  enabled          = true
  servers          = ["127.0.0.1:4647"]
  max_kill_timeout = "10m" // increased from default to accomodate ECS.

  host_volume "input" {
  path        = "/Users/input/data/ckatar/isic_2018"
  read_only   = false
}
  host_volume "output" {
  path        = "/Users/output/ckatar/isic_2018/modelsave"
  read_only   = false
}


  }