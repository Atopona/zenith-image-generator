const STORAGE_KEY = 'z-image-api-key-encrypted'

async function getKey(): Promise<CryptoKey> {
  const fingerprint = [navigator.userAgent, navigator.language, screen.width, screen.height].join('|')
  const encoder = new TextEncoder()
  const keyMaterial = await crypto.subtle.importKey('raw', encoder.encode(fingerprint), 'PBKDF2', false, ['deriveKey'])
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: encoder.encode('z-image-salt'), iterations: 100000, hash: 'SHA-256' },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  )
}

export async function encryptAndStore(apiKey: string): Promise<void> {
  if (!apiKey) {
    localStorage.removeItem(STORAGE_KEY)
    return
  }
  const key = await getKey()
  const iv = crypto.getRandomValues(new Uint8Array(12))
  const encrypted = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, new TextEncoder().encode(apiKey))
  const data = JSON.stringify({ iv: Array.from(iv), data: Array.from(new Uint8Array(encrypted)) })
  localStorage.setItem(STORAGE_KEY, data)
}

export async function decryptFromStore(): Promise<string> {
  const stored = localStorage.getItem(STORAGE_KEY)
  if (!stored) return ''
  try {
    const { iv, data } = JSON.parse(stored)
    const key = await getKey()
    const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: new Uint8Array(iv) }, key, new Uint8Array(data))
    return new TextDecoder().decode(decrypted)
  } catch {
    localStorage.removeItem(STORAGE_KEY)
    return ''
  }
}
