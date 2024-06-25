
log_level  = "DEBUG"
datacenter = "dc1"

data_dir = "/tmp/local_client"
name     = "local_client1"

plugin "raw_exec" {
    config {
    enabled    = true
    no_cgroups = true
  }
 options {
        "driver.allowlist" = "exec,java,raw_exec"
      }
}

client {
    enabled          = true
    servers          = ["127.0.0.1:4647"]
    max_kill_timeout = "10m" // increased from default to accomodate ECS.

    host_volume "input" {
    path        = "/Users/input"
    read_only   = false
    }
    host_volume "output" {
    path        = "/Users/output"
    read_only   = false
    }

}

ports {
  http = 5656
  rpc  = 5657
  serf = 5658
}
