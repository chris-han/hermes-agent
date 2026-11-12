import type { Translations } from './types'

const cloudSignInErrors: Record<string, string> = {
  cloud_request_timeout: 'The Cloud request timed out. Retry the same connection; authorization is retained.',
  cloud_metadata_timeout: 'Model capability lookup timed out. Retry model discovery; your device remains connected.',
  cloud_metadata_unavailable: 'Model capability information is unavailable. Retry discovery.',
  cloud_model_access_missing: 'No model access is enabled for this account. Ask your account administrator to check access.',
  cloud_provisioning_pending: 'Model access is being prepared. Retry readiness shortly.',
  cloud_entitlement_restricted: 'This account cannot currently use the selected model. Review account access and credit.',
  cloud_capability_mismatch: 'This model capability revision is unsupported. Refresh models or update the app.',
  cloud_catalog_timeout: 'Model discovery timed out. Retry discovery; your device remains connected.',
  cloud_catalog_empty: 'This account has no enabled model. Refresh models after access is assigned. Your device remains connected.',
  cloud_catalog_unavailable: 'Model discovery is temporarily unavailable. Retry discovery; your device remains connected.',
  cloud_network_unavailable: 'Semantier Cloud could not be reached. Check your network connection and retry.',
  cloud_dns_failed: 'Semantier Cloud DNS lookup failed. Check DNS and the configured Cloud provider, then retry.',
  cloud_tls_failed:
    'Semantier Cloud could not validate its TLS certificate. Check the provider and trusted CA settings.',
  cloud_http3_unavailable:
    'Semantier Cloud HTTP/3 is unavailable. Check HTTP/3 support and UDP connectivity, then retry.',
  cloud_config_invalid:
    'Semantier Cloud provider settings are invalid. Check this profile’s configuration or use an API key.',
  cloud_response_invalid: 'Semantier Cloud returned an invalid response. Retry the same connection.',
  cloud_profile_mismatch:
    'This sign-in does not match profile {profile}. Return to the original profile before retrying.',
  cloud_profile_changed: 'Profile {profile} changed during sign-in. Return to that profile and start sign-in again.',
  cloud_profile_unavailable: 'Profile {profile} is unavailable. Return to the original profile before retrying.',
  cloud_storage_invalid:
    'Protected Cloud credentials for profile {profile} could not be read. Reauthorize in that profile.',
  cloud_storage_unavailable:
    'Windows protected credential storage is unavailable. Check the Windows user session or use an API key.',
  cloud_credential_expired: 'Your Semantier Cloud authorization expired. Start sign-in again in profile {profile}.',
  cloud_authorization_required: 'Semantier Cloud requires authorization. Start sign-in again in this profile.',
  cloud_authorization_denied:
    'Semantier Cloud refused authorization. Check your account and permissions before retrying.',
  cloud_server_conflict: 'Semantier Cloud reported a server-side conflict. Retry the same connection.',
  cloud_session_unavailable:
    'This sign-in session is unavailable in profile {profile}. Return to its original profile or start again.',
  cloud_session_expired: 'This sign-in session expired. Try again to open a new authorization page.',
  cloud_session_cancelled: 'This sign-in was cancelled. Try again when you are ready.',
  cloud_rate_limited: 'Semantier Cloud is busy. Wait briefly, then try again.',
  cloud_origin_redirected: 'Semantier Cloud redirected sign-in. Check the configured service origin.',
  cloud_protocol_unsupported: 'Cloud protocol unsupported by this Hermes version. Update the app or check the configured service.',
  cloud_server_error: 'Semantier Cloud could not complete sign-in. Retry later or use an API key.',
  cloud_internal_error: 'Semantier Cloud sign-in could not complete. Try again or use an API key.',
  cloud_unknown_failure: 'Semantier Cloud sign-in did not finish. Try again or use an API key.'
}

export const enOnboarding: Translations['onboarding'] = {
  headerTitle: "Let's get you setup with Semantier Agent",
  headerDesc: 'Connect a model provider to start chatting. Most options take one click.',
  preparingInstall: 'Semantier is finishing install. This usually takes under a minute on first run.',
  starting: 'Starting Semantier…',
  setupSlowTitle: 'Setup is taking longer than usual.',
  setupSlowBody: 'Semantier is still starting in the background.',
  continueWithoutSetup: 'Continue without setup',
  lookingUpProviders: 'Looking up providers...',
  collapse: 'Collapse',
  otherProviders: 'Other providers',
  haveApiKey: 'I have an API key',
  chooseLater: "I'll choose a provider later",
  recommended: 'Recommended',
  connected: 'Connected',
  featuredPitch: 'One subscription, 300+ frontier models — the recommended way to run Semantier',
  fireworksPitch: 'Direct model API — Fireworks-hosted frontier models',
  openRouterPitch: 'One key, hundreds of models — a solid default',
  apiKeyOptions: {
    fireworks: {
      short: 'direct model API',
      description: 'Direct access to models hosted by Fireworks AI.'
    },
    openrouter: {
      short: 'one key, many models',
      description: 'Hosts hundreds of models behind a single key. Good default for new installs.'
    },
    openai: { short: 'GPT-class models', description: 'Direct access to OpenAI models.' },
    gemini: { short: 'Gemini models', description: 'Direct access to Google Gemini models.' },
    xai: { short: 'Grok models', description: 'Direct access to xAI Grok models.' },
    local: {
      short: 'self-hosted',
      description:
        'Point Semantier at a local or self-hosted OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, etc).'
    }
  },
  backToSignIn: 'Back to sign in',
  getKey: 'Get a key',
  replaceCurrent: 'Replace current value',
  pasteApiKey: 'Paste API key',
  localApiKeyPlaceholder: 'API key (optional — only if your endpoint requires one)',
  localModelNamePlaceholder: 'Model name (e.g. command-a-plus-05-2026)',
  couldNotSave: 'Could not save credential.',
  connecting: 'Connecting',
  update: 'Update',
  flowSubtitles: {
    pkce: 'Opens your browser to sign in, then continues here',
    device_code: 'Opens a verification page in your browser — Semantier connects automatically',
    external: 'Sign in once in your terminal, then come back to chat'
  },
  startingSignIn: provider => `Starting sign-in for ${provider}...`,
  verifyingCode: provider => `Verifying your code with ${provider}...`,
  connectedProvider: provider => `${provider} connected`,
  connectedPicking: provider => `${provider} connected. Picking a default model...`,
  cloudSignInError: (code, profile) =>
    (cloudSignInErrors[code] || cloudSignInErrors.cloud_unknown_failure).replace('{profile}', profile),
  signInFailed: 'Sign-in failed. Try again.',
  signInExpired:
    'The sign-in page timed out before you finished. Try again and complete the browser step within a few minutes, or use an API key instead.',
  signInDidNotFinish: provider =>
    `Sign-in with ${provider} did not finish. Check your internet connection and try again, or pick a different provider.`,
  tryAgain: 'Try again',
  useApiKeyInstead: 'Use an API key',
  errorDetails: 'Details',
  pickDifferentProvider: 'Pick a different provider',
  signInWith: provider => `Sign in with ${provider}`,
  openedBrowser: provider => `We opened ${provider} in your browser.`,
  authorizeThere: 'Authorize Hermes there.',
  copyAuthCode: 'Copy the authorization code and paste it below.',
  pasteAuthCode: 'Paste authorization code',
  reopenAuthPage: 'Re-open authorization page',
  waitingAuthorize: 'Waiting for you to authorize...',
  externalPending: provider =>
    `${provider} signs in through its own CLI. Run this command in a terminal, then come back and pick "I've signed in":`,
  signedIn: "I've signed in",
  cloudBrowserOpened: 'Approve this device in your browser. Return to this profile to finish connecting.',
  deviceCodeOpened: provider => `We opened ${provider} in your browser. Enter this code there:`,
  reopenVerification: 'Re-open verification page',
  copy: 'Copy',
  defaultModel: 'Default model',
  freeTier: 'Free tier',
  pro: 'Pro',
  free: 'Free',
  price: (input, output) => `${input} in / ${output} out per Mtok`,
  change: 'Change',
  startChatting: 'Begin',
  docs: provider => `${provider} docs`
}
