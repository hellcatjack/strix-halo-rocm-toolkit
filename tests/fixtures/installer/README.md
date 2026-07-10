# Installer fixtures

These fixtures drive the explicitly enabled installer test backend. They never
invoke sudo, Docker, package managers, or GPU devices. Production CLI routing
rejects the fixture environment unless `AMD_AI_INSTALLER_ENABLE_FIXTURES=1` is
also present.

`boot_id` is mutable test evidence. A full-mode run remains at
`REBOOT_PENDING` until its canonical UUID changes.
