~/MINI-V_Android/vendor/mini-v/ai/
├── etc/
│   └── init.mini-v-ai.rc          ← Step 1
├── daemon/
│   ├── Android.bp                 ← Step 2 -> for build
│   └── main.cpp                   ← Step 2 -> demon skeleton
├── sepolicy/
│   └── vendor/
│       ├── file_contexts          ← Step 3
│       ├── service_contexts       ← Step 3 -> for change to hal
│       └── miniv_ai.te            ← Step 3
├── cli/
│   ├── Android.bp                 ← Step 4
│   └── main.cpp                   ← Step 4
└── Android.bp                     ← route, cc_defualt define