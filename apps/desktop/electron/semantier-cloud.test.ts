import { expect, test } from 'vitest'
import { runGatewayEnsure } from './local-gateway'

test.each([undefined, '0', '1'])('gateway child receives explicit cloud selection %s', async value => {
  const parent = { PATH: process.env.PATH ?? '', ...(value === undefined ? {} : { SEMANTIER_CLOUD: value }) }
  const result = await runGatewayEnsure({
    command: process.execPath,
    args: ['-e', 'process.stdout.write(JSON.stringify({mode:process.env.SEMANTIER_CLOUD, provider:process.env.PROVIDER}))'],
    env: { PROVIDER: 'independent' }, shell: false,
  }, process.cwd(), '/fixture/home', parent)
  expect(JSON.parse(result.stdout)).toEqual({mode:value ?? '0', provider:'independent'})
})

test('invalid or contradictory mode inputs fail before a child is launched', async () => {
  const backend = {command:process.execPath, args:['-e', 'process.exit(99)'], env:{}, shell:false}
  await expect(runGatewayEnsure(backend, process.cwd(), '/fixture/home', {SEMANTIER_CLOUD:'true'})).rejects.toThrow('SEMANTIER_CLOUD')
  await expect(runGatewayEnsure({...backend, env:{SEMANTIER_CLOUD:'1'}}, process.cwd(), '/fixture/home', {SEMANTIER_CLOUD:'0'})).rejects.toThrow('SEMANTIER_CLOUD')
})
