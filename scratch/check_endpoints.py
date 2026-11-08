import asyncio
from asyncua import Client

async def main():
    url = "opc.tcp://127.0.0.1:48410/brewscada/server/"
    client = Client(url=url)
    endpoints = await client.connect_and_get_server_endpoints()
    print("Endpoints returned by server:")
    for ep in endpoints:
        print(f"  EndpointURL: {ep.EndpointUrl}")
        print(f"  SecurityMode: {ep.SecurityMode}")
        print(f"  SecurityPolicyUri: {ep.SecurityPolicyUri}")

asyncio.run(main())
