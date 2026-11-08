import asyncio
from asyncua import Client

async def main():
    url = "opc.tcp://127.0.0.1:48410/brewscada/server/"
    print(f"Connecting to {url}...")
    async with Client(url=url) as client:
        ns_array = await client.get_namespace_array()
        print("Namespace Array:")
        for idx, ns in enumerate(ns_array):
            print(f"  [{idx}] = {ns}")

        root = client.nodes.objects
        children = await root.get_children()
        print("\nObjects children:")
        for child in children:
            name = await child.read_browse_name()
            node_id = child.nodeid
            print(f"  BrowseName: {name}, NodeId: {node_id}")
            if "BrewSCADA" in str(node_id):
                sub_children = await child.get_children()
                for sub in sub_children:
                    sub_name = await sub.read_browse_name()
                    print(f"    -> Sub BrowseName: {sub_name}, NodeId: {sub.nodeid}")
                    tags = await sub.get_children()
                    for t in tags[:5]:
                        t_name = await t.read_browse_name()
                        val = await t.read_value()
                        print(f"       * Tag {t_name}: NodeId={t.nodeid}, Value={val}")

asyncio.run(main())
