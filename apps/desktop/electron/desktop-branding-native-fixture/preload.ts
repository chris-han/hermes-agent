import { contextBridge } from 'electron'

import { createBrandingBridge } from '../desktop-branding-bridge'
contextBridge.exposeInMainWorld('hermesDesktop', { branding: createBrandingBridge() })
