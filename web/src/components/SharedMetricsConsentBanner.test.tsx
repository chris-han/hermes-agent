// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { I18nProvider } from "@/i18n";
import { SharedMetricsConsentBanner } from "./SharedMetricsConsentBanner";

it("names the sharing recipient and does not link an upstream privacy policy", async () => {
  sessionStorage.clear();
  vi.spyOn(api, "getSharedMetricsConsent").mockResolvedValue({ enabled: false, send: false, decided: false, reask: false, managed: false });
  const save = vi.spyOn(api, "saveSharedMetricsConsent").mockResolvedValue({ enabled: true, send: true, decided: true, reask: false, managed: false });
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  try {
    await act(async () => root.render(<I18nProvider><SharedMetricsConsentBanner /></I18nProvider>));
    const share = [...container.querySelectorAll("button")].find(button => button.textContent === "Send to Semantier");
    expect(share).toBeTruthy();
    expect(container.textContent).toContain("sending to Semantier is a separate choice");
    expect(container.querySelector("a")).toBeNull();
    await act(async () => share!.click());
    expect(save).toHaveBeenCalledWith({ enabled: true, send: true });
  } finally {
    await act(async () => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  }
});
